import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from . import utils as lora_utils
from .evaluation import build_stop_token_sequences, prepare_model, trim_stop_sequences


@dataclass(frozen=True)
class ModelPreset:
    label: str
    model: str
    adapter_file: str | None = None
    lora_layers: int = 4


@dataclass(frozen=True)
class Suggestion:
    question: str
    source_reference: str = ""
    source_doc: str = ""


@dataclass(frozen=True)
class MatrixVariant:
    label: str
    preset_key: str
    preset: ModelPreset
    condition: str


MODEL_PRESETS = {
    "tinyllama-clean": ModelPreset(
        label="TinyLlama Pasal.id clean adapter",
        model="mlx_model",
        adapter_file="outputs/adapters/adapters_pasalid_tinyllama_native_expanded_clean.npz",
    ),
    "tinyllama-base": ModelPreset(
        label="TinyLlama base",
        model="mlx_model",
    ),
    "tinyllama-json-large": ModelPreset(
        label="TinyLlama Pasal.id JSON-large adapter",
        model="mlx_model",
        adapter_file="outputs/adapters/adapters_pasalid_tinyllama_experiment.npz",
    ),
    "mistral-q4-long": ModelPreset(
        label="Mistral q4 Pasal.id long adapter",
        model="mlx_model_mistral_q4",
        adapter_file="outputs/adapters/adapters_pasalid_mistral_q4_experiment_long.npz",
    ),
    "mistral-q4-base": ModelPreset(
        label="Mistral q4 base",
        model="mlx_model_mistral_q4",
    ),
    "qwen3": ModelPreset(
        label="Qwen3 Pasal.id adapter",
        model="mlx-community/Qwen3-4B-8bit",
        adapter_file="outputs/adapters/adapters_pasalid_qwen3_experiment.npz",
    ),
}


CONDITIONS = {
    "A": "base, no source context",
    "B": "base, with source context",
    "C": "adapter, no source context",
    "D": "adapter, with source context",
}


DEFAULT_STOP_STRINGS = ["\nQ:", "\nA:", "table:", "columns:"]
DEFAULT_SUGGESTIONS_FILE = Path("data/pasalid/realcase_eval.jsonl")
DEFAULT_MATRIX_PRESETS = ["tinyllama-clean", "mistral-q4-long", "qwen3"]
DEFAULT_MATRIX_CONDITIONS = ["A", "B", "C", "D"]

DEFAULT_SUGGESTIONS = [
    Suggestion("Apa aturan hukum jika seseorang menyerang kehormatan atau nama baik orang lain melalui sistem elektronik?", "UU No. 1 Tahun 2024, Pasal 27A"),
    Suggestion("Apa aturan tentang pengiriman informasi elektronik yang berisi ancaman kekerasan kepada korban?", "UU No. 1 Tahun 2024, Pasal 29"),
    Suggestion("Apa sanksi hukum jika seseorang mengirim ancaman kekerasan atau pesan yang menakut-nakuti korban melalui media elektronik?", "UU No. 1 Tahun 2024, Pasal 45B"),
    Suggestion("Apa yang dimaksud dengan Provinsi Kepulauan Bangka Belitung dalam aturan ini?", "UU No. 31 Tahun 2024, Pasal 1"),
    Suggestion("Berapa jumlah kecamatan di Kabupaten Belitung dan apa saja kecamatannya?", "UU No. 31 Tahun 2024, Pasal 3"),
    Suggestion("Apa karakteristik wilayah dan potensi utama Kabupaten Belitung?", "UU No. 31 Tahun 2024, Pasal 6"),
    Suggestion("Apa dasar pengaturan susunan dan tata cara penyelenggaraan pemerintahan daerah?", "UU No. 31 Tahun 2024, Pasal 7"),
    Suggestion("Apa aturan mengenai anggaran kementerian dan lembaga dalam APBN?", "UU No. 62 Tahun 2024"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive terminal chat for LoRA/MLX models.")
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), help="Model preset to load")
    parser.add_argument("--model", help="Custom local MLX model path or HF repo")
    parser.add_argument("--adapter-file", help="Custom LoRA adapter file")
    parser.add_argument("--lora-layers", type=int, default=4, help="Number of last layers wrapped with LoRA")
    parser.add_argument("--condition", choices=sorted(CONDITIONS), help="A/B/C/D experiment condition")
    parser.add_argument("--context-file", help="Optional source context file for B/D prompts")
    parser.add_argument("--suggestions-file", help="Optional question suggestions file: txt, JSONL, or JSON array. Defaults to data/pasalid/realcase_eval.jsonl when available")
    parser.add_argument("--reference", default="", help="Optional source reference line for B/D prompts")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens per turn")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--no-stream", action="store_true", help="Print after full generation instead of streaming")
    parser.add_argument("--stop-strings", nargs="*", default=DEFAULT_STOP_STRINGS, help="Stop generation strings")
    parser.add_argument(
        "--matrix",
        choices=["conditions", "models", "all"],
        help="Compare answers in one interface: conditions=A/B/C/D for one model, models=same condition across models, all=models x conditions",
    )
    parser.add_argument(
        "--matrix-presets",
        nargs="*",
        choices=sorted(MODEL_PRESETS),
        help="Model presets for --matrix models/all. Defaults to TinyLlama clean, Mistral q4 long, and Qwen3.",
    )
    parser.add_argument(
        "--matrix-conditions",
        nargs="*",
        choices=sorted(CONDITIONS),
        help="Conditions for --matrix conditions/all. Defaults to A B C D.",
    )
    return parser


def choose_from_menu(title: str, options: list[tuple[str, str]], default: str | None = None) -> str:
    print(title)
    for index, (key, label) in enumerate(options, start=1):
        default_marker = " [default]" if key == default else ""
        print(f"  {index}. {key} - {label}{default_marker}")

    while True:
        choice = input("> ").strip()
        if not choice and default is not None:
            return default
        if choice in {key for key, _ in options}:
            return choice
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(options):
                return options[index][0]
        print("Pilih nomor atau key yang tersedia.")


def choose_many_from_menu(
    title: str,
    options: list[tuple[str, str]],
    default: list[str],
) -> list[str]:
    print(title)
    print("Pisahkan pilihan dengan koma/spasi, atau tekan enter untuk default.")
    default_set = set(default)
    for index, (key, label) in enumerate(options, start=1):
        default_marker = " [default]" if key in default_set else ""
        print(f"  {index}. {key} - {label}{default_marker}")

    keys = {key for key, _ in options}
    while True:
        choice = input("> ").strip()
        if not choice:
            return default
        selected = []
        for part in choice.replace(",", " ").split():
            if part in keys:
                selected.append(part)
            elif part.isdigit():
                index = int(part) - 1
                if 0 <= index < len(options):
                    selected.append(options[index][0])
        unique = list(dict.fromkeys(selected))
        if unique:
            return unique
        print("Pilih nomor atau key yang tersedia.")


def resolve_preset(args: argparse.Namespace) -> ModelPreset:
    if args.model:
        return ModelPreset(
            label="custom",
            model=args.model,
            adapter_file=args.adapter_file,
            lora_layers=args.lora_layers,
        )

    preset_key = args.preset
    if not preset_key:
        preset_key = choose_from_menu(
            "Pilih model:",
            [(key, preset.label) for key, preset in MODEL_PRESETS.items()],
            default="tinyllama-clean",
        )
    return MODEL_PRESETS[preset_key]


def resolve_condition(args: argparse.Namespace, preset: ModelPreset) -> str:
    condition = args.condition
    if condition:
        return condition
    default = "D" if preset.adapter_file else "B"
    return choose_from_menu(
        "Pilih mode eksperimen:",
        [(key, label) for key, label in CONDITIONS.items()],
        default=default,
    )


def resolve_matrix_presets(args: argparse.Namespace) -> list[str]:
    if args.matrix_presets:
        return args.matrix_presets
    if args.matrix == "conditions" and args.preset:
        return [args.preset]
    if args.matrix == "conditions":
        return ["tinyllama-clean"]
    return DEFAULT_MATRIX_PRESETS


def resolve_matrix_conditions(args: argparse.Namespace) -> list[str]:
    if args.matrix_conditions:
        return args.matrix_conditions
    if args.matrix == "models" and args.condition:
        return [args.condition]
    if args.matrix == "models":
        return ["D"]
    return DEFAULT_MATRIX_CONDITIONS


def build_matrix_variants(args: argparse.Namespace) -> list[MatrixVariant]:
    if args.model:
        if args.matrix in {"models", "all"}:
            raise ValueError("--model custom hanya mendukung --matrix conditions. Gunakan --preset untuk matrix antar model.")
        preset = ModelPreset("custom", args.model, args.adapter_file, args.lora_layers)
        variants = []
        for condition in resolve_matrix_conditions(args):
            if needs_adapter(condition) and not preset.adapter_file:
                continue
            variants.append(MatrixVariant(f"custom/{condition}", "custom", preset, condition))
        return variants

    preset_keys = resolve_matrix_presets(args)
    conditions = resolve_matrix_conditions(args)
    if args.matrix == "models":
        conditions = conditions[:1]

    variants = []
    for preset_key in preset_keys:
        preset = MODEL_PRESETS[preset_key]
        for condition in conditions:
            if needs_adapter(condition) and not preset.adapter_file:
                continue
            label = f"{preset_key}/{condition}"
            variants.append(MatrixVariant(label, preset_key, preset, condition))
    if not variants:
        raise ValueError("Tidak ada matrix variant valid. Pastikan preset yang dipilih punya adapter untuk mode C/D.")
    return variants


def build_matrix_variants_from_choices(
    mode: str,
    preset_keys: list[str],
    conditions: list[str],
) -> list[MatrixVariant]:
    if mode == "models":
        conditions = conditions[:1]
    variants = []
    for preset_key in preset_keys:
        preset = MODEL_PRESETS[preset_key]
        for condition in conditions:
            if needs_adapter(condition) and not preset.adapter_file:
                continue
            variants.append(MatrixVariant(f"{preset_key}/{condition}", preset_key, preset, condition))
    return variants


def configure_matrix_interactive() -> tuple[str, list[MatrixVariant]]:
    mode = choose_from_menu(
        "Pilih mode matrix:",
        [
            ("conditions", "bandingkan A/B/C/D untuk satu model"),
            ("models", "bandingkan satu condition lintas model"),
            ("all", "bandingkan model x condition"),
            ("off", "matikan matrix mode"),
        ],
        default="conditions",
    )
    if mode == "off":
        return mode, []

    if mode == "conditions":
        preset_key = choose_from_menu(
            "Pilih model untuk matrix conditions:",
            [(key, preset.label) for key, preset in MODEL_PRESETS.items()],
            default="tinyllama-clean",
        )
        preset_keys = [preset_key]
        conditions = choose_many_from_menu(
            "Pilih conditions:",
            [(key, label) for key, label in CONDITIONS.items()],
            default=DEFAULT_MATRIX_CONDITIONS,
        )
    elif mode == "models":
        preset_keys = choose_many_from_menu(
            "Pilih model:",
            [(key, preset.label) for key, preset in MODEL_PRESETS.items()],
            default=DEFAULT_MATRIX_PRESETS,
        )
        condition = choose_from_menu(
            "Pilih condition yang dibandingkan:",
            [(key, label) for key, label in CONDITIONS.items()],
            default="D",
        )
        conditions = [condition]
    else:
        preset_keys = choose_many_from_menu(
            "Pilih model:",
            [(key, preset.label) for key, preset in MODEL_PRESETS.items()],
            default=DEFAULT_MATRIX_PRESETS,
        )
        conditions = choose_many_from_menu(
            "Pilih conditions:",
            [(key, label) for key, label in CONDITIONS.items()],
            default=["B", "D"],
        )

    variants = build_matrix_variants_from_choices(mode, preset_keys, conditions)
    if not variants:
        print("Tidak ada variant valid. Preset tanpa adapter tidak bisa dipakai untuk C/D.")
        return "off", []
    return mode, variants


def needs_adapter(condition: str) -> bool:
    return condition in {"C", "D"}


def needs_context(condition: str) -> bool:
    return condition in {"B", "D"}


def read_context(context_file: str | None) -> str:
    if not context_file:
        return ""
    return Path(context_file).read_text().strip()


def extract_question(text: str) -> str | None:
    if "Q: " not in text:
        return None
    question = text.split("Q: ", 1)[1].split("\nA: ", 1)[0].strip()
    return question or None


def extract_context(text: str) -> str:
    if not text.startswith("Dokumen sumber: ") or "\nReferensi: " not in text:
        return ""
    return text.split("Dokumen sumber: ", 1)[1].split("\nReferensi: ", 1)[0].strip()


def extract_reference(text: str) -> str:
    if "\nReferensi: " not in text or "\nQ: " not in text:
        return ""
    return text.split("\nReferensi: ", 1)[1].split("\nQ: ", 1)[0].strip()


def suggestion_from_row(row: dict) -> Suggestion | None:
    if isinstance(row.get("question"), str):
        return Suggestion(
            question=row["question"].strip(),
            source_reference=str(row.get("source_reference", "")).strip(),
            source_doc=str(row.get("source_doc", "")).strip(),
        )
    if isinstance(row.get("text"), str):
        question = extract_question(row["text"])
        if question:
            return Suggestion(
                question=question,
                source_reference=extract_reference(row["text"]),
                source_doc=extract_context(row["text"]),
            )
    return None


def read_suggestions(suggestions_file: str | None) -> list[Suggestion]:
    path = Path(suggestions_file) if suggestions_file else DEFAULT_SUGGESTIONS_FILE
    if not path.exists():
        return DEFAULT_SUGGESTIONS

    content = path.read_text().strip()
    if not content:
        return DEFAULT_SUGGESTIONS

    suggestions = []
    if content.startswith("["):
        values = json.loads(content)
        for value in values:
            if isinstance(value, str):
                suggestions.append(Suggestion(value.strip()))
            elif isinstance(value, dict):
                suggestion = suggestion_from_row(value)
                if suggestion:
                    suggestions.append(suggestion)
    else:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                suggestion = suggestion_from_row(row)
                if suggestion:
                    suggestions.append(suggestion)
            else:
                suggestions.append(Suggestion(line))

    unique = []
    seen = set()
    for suggestion in suggestions:
        if not suggestion.question or suggestion.question in seen:
            continue
        seen.add(suggestion.question)
        unique.append(suggestion)
    return unique or DEFAULT_SUGGESTIONS


def paste_multiline(prompt: str) -> str:
    print(prompt)
    print("Akhiri dengan baris kosong.")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def build_prompt(question: str, condition: str, context: str, reference: str) -> str:
    if needs_context(condition) and context:
        chunks = [f"Dokumen sumber: {context}"]
        if reference:
            chunks.append(f"Referensi: {reference}")
        chunks.append(f"Q: {question}")
        chunks.append("A: ")
        return "\n".join(chunks)
    return f"Q: {question}\nA: "


def print_suggestions(suggestions: list[Suggestion], limit: int = 10) -> None:
    print("Suggested questions:")
    for index, suggestion in enumerate(suggestions[:limit], start=1):
        source = f" [{suggestion.source_reference}]" if suggestion.source_reference else ""
        print(f"  {index}. {suggestion.question}{source}")
    print("Ketik nomor untuk langsung memakai pertanyaan, atau /suggest N untuk jumlah lain.")


def generate_reply(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temp: float,
    stop_strings: list[str],
    stop_token_sequences: list[tuple[str, list[int]]],
    stream: bool,
) -> str:
    prompt_ids = mx.array(tokenizer.encode(prompt))
    tokens = []
    printed = 0

    for token, _ in zip(lora_utils.generate(prompt_ids, model, temp=temp), range(max_new_tokens)):
        token_id = token.item()
        if token_id == tokenizer.eos_token_id:
            break
        tokens.append(token_id)

        for _, stop_token_ids in stop_token_sequences:
            if len(tokens) >= len(stop_token_ids) and tokens[-len(stop_token_ids) :] == stop_token_ids:
                tokens = tokens[: -len(stop_token_ids)]
                text = tokenizer.decode(tokens)
                if stream:
                    remainder = text[printed:]
                    if remainder:
                        print(remainder, end="", flush=True)
                    print()
                return text.strip()

        if stream:
            text = tokenizer.decode(tokens)
            if len(text) - printed > 1:
                print(text[printed:-1], end="", flush=True)
                printed = len(text) - 1

    text = trim_stop_sequences(tokenizer.decode(tokens), stop_strings)
    if stream:
        remainder = text[printed:]
        if remainder:
            print(remainder, end="", flush=True)
        print()
    return text


def run_matrix(
    variants: list[MatrixVariant],
    question: str,
    context: str,
    reference: str,
    max_new_tokens: int,
    temp: float,
    stop_strings: list[str],
) -> None:
    print("Matrix answers:")
    for variant in variants:
        adapter_file = variant.preset.adapter_file if needs_adapter(variant.condition) else None
        print(f"\n=== {variant.label}: {CONDITIONS[variant.condition]} ===")
        print("Loading variant...", flush=True)
        model, tokenizer = prepare_model(variant.preset.model, adapter_file, variant.preset.lora_layers)
        stop_token_sequences = build_stop_token_sequences(tokenizer, stop_strings)
        prompt = build_prompt(question, variant.condition, context, reference)
        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temp=temp,
            stop_strings=stop_strings,
            stop_token_sequences=stop_token_sequences,
            stream=False,
        )
        print(reply or "(empty)")


def print_help() -> None:
    print("Commands:")
    print("  /help       tampilkan bantuan")
    print("  /context    paste dokumen sumber untuk mode B/D")
    print("  /matrix     konfigurasi matrix compare interaktif")
    print("  /single     matikan matrix compare")
    print("  /suggest    tampilkan contoh pertanyaan")
    print("  /clear      hapus dokumen sumber")
    print("  /show       tampilkan status model/mode/context")
    print("  /exit       keluar")
    print("Ketik pertanyaan langsung untuk mendapat jawaban.")


def main() -> None:
    args = build_parser().parse_args()
    matrix_mode = args.matrix or ""
    matrix_variants = build_matrix_variants(args) if args.matrix else []
    preset = resolve_preset(args) if not args.matrix else None
    condition = resolve_condition(args, preset) if preset else None

    adapter_file = preset.adapter_file if preset and needs_adapter(condition) else None
    if condition and needs_adapter(condition) and not adapter_file:
        raise ValueError(f"Mode {condition} membutuhkan adapter, tetapi preset ini tidak punya adapter_file.")

    if matrix_mode:
        print(f"Matrix mode: {matrix_mode}")
        print("Variants:")
        for variant in matrix_variants:
            print(f"  - {variant.label}: {CONDITIONS[variant.condition]}")
        model = None
        tokenizer = None
        stop_token_sequences = []
    else:
        print(f"Loading model: {preset.label}")
        print(f"Mode: {condition} - {CONDITIONS[condition]}")
        model, tokenizer = prepare_model(preset.model, adapter_file, preset.lora_layers)
        stop_token_sequences = build_stop_token_sequences(tokenizer, args.stop_strings)
    context = read_context(args.context_file)
    suggestions = read_suggestions(args.suggestions_file)
    reference = args.reference.strip()

    matrix_needs_context = bool(matrix_variants) and any(needs_context(variant.condition) for variant in matrix_variants)
    if ((condition and needs_context(condition)) or matrix_needs_context) and not context:
        print("Mode ini memakai dokumen sumber. Gunakan /context untuk paste konteks, atau tetap lanjut tanpa konteks.")
    print_help()
    print_suggestions(suggestions, limit=5)

    while True:
        try:
            user_input = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            break
        if user_input == "/help":
            print_help()
            continue
        if user_input == "/context":
            context = paste_multiline("Paste dokumen sumber:")
            continue
        if user_input == "/matrix":
            matrix_mode, matrix_variants = configure_matrix_interactive()
            if matrix_variants:
                print(f"Matrix mode aktif: {matrix_mode}")
                for variant in matrix_variants:
                    print(f"  - {variant.label}: {CONDITIONS[variant.condition]}")
            else:
                print("Matrix mode off.")
            continue
        if user_input == "/single":
            matrix_mode = ""
            matrix_variants = []
            print("Matrix mode off. Kembali ke single answer mode.")
            continue
        if user_input.startswith("/suggest"):
            parts = user_input.split()
            limit = 10
            if len(parts) > 1 and parts[1].isdigit():
                limit = max(1, int(parts[1]))
            print_suggestions(suggestions, limit=limit)
            continue
        if user_input == "/clear":
            context = ""
            print("Context cleared.")
            continue
        if user_input == "/show":
            if matrix_variants:
                print(f"Matrix mode: {matrix_mode}")
                print("Variants:")
                for variant in matrix_variants:
                    print(f"  - {variant.label}: {CONDITIONS[variant.condition]}")
            else:
                print(f"Model: {preset.label}")
                print(f"Mode: {condition} - {CONDITIONS[condition]}")
                print(f"Adapter: {adapter_file or '-'}")
            print(f"Context chars: {len(context)}")
            print(f"Suggestions: {len(suggestions)}")
            print(f"Default suggestions file: {DEFAULT_SUGGESTIONS_FILE}")
            continue

        if user_input.isdigit():
            suggestion_index = int(user_input) - 1
            if 0 <= suggestion_index < len(suggestions):
                suggestion = suggestions[suggestion_index]
                user_input = suggestion.question
                if suggestion.source_doc:
                    context = suggestion.source_doc
                if suggestion.source_reference:
                    reference = suggestion.source_reference
                print(f"Q: {user_input}")
                if suggestion.source_reference:
                    print(f"Source: {suggestion.source_reference}")
            else:
                print("Nomor suggestion tidak tersedia. Pakai /suggest untuk melihat daftar.")
                continue

        if matrix_variants:
            run_matrix(
                variants=matrix_variants,
                question=user_input,
                context=context,
                reference=reference,
                max_new_tokens=args.max_new_tokens,
                temp=args.temp,
                stop_strings=args.stop_strings,
            )
        else:
            prompt = build_prompt(user_input, condition, context, reference)
            print("A> ", end="", flush=True)
            reply = generate_reply(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temp=args.temp,
                stop_strings=args.stop_strings,
                stop_token_sequences=stop_token_sequences,
                stream=not args.no_stream,
            )
            if args.no_stream:
                print(reply)


if __name__ == "__main__":
    main()
