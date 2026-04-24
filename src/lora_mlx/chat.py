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

DEFAULT_SUGGESTIONS = [
    "Apa dasar hukum monitoring dan evaluasi kementerian dan lembaga?",
    "Apa dasar hukum pembentukan peraturan daerah oleh pemerintah daerah?",
    "Apa kewajiban pemerintah daerah dalam menyusun rencana pembangunan daerah?",
    "Apa dasar hukum pengadaan barang dan jasa pemerintah?",
    "Apa dasar hukum perlindungan data pribadi di Indonesia?",
    "Apa kewenangan pemerintah pusat dalam pembinaan dan pengawasan pemerintah daerah?",
    "Apa sanksi bagi penyelenggara negara yang tidak melaporkan harta kekayaan?",
    "Apa dasar hukum keterbukaan informasi publik?",
    "Apa kewajiban badan publik dalam menyediakan informasi kepada masyarakat?",
    "Apa dasar hukum evaluasi akuntabilitas kinerja instansi pemerintah?",
    "Apa dasar hukum sistem pemerintahan berbasis elektronik?",
    "Apa kewajiban kementerian atau lembaga dalam penyusunan laporan kinerja?",
    "Apa dasar hukum pengelolaan keuangan negara?",
    "Apa perbedaan kewenangan pemerintah pusat dan pemerintah daerah dalam urusan pemerintahan?",
    "Apa dasar hukum pengawasan internal pemerintah oleh APIP?",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive terminal chat for LoRA/MLX models.")
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), help="Model preset to load")
    parser.add_argument("--model", help="Custom local MLX model path or HF repo")
    parser.add_argument("--adapter-file", help="Custom LoRA adapter file")
    parser.add_argument("--lora-layers", type=int, default=4, help="Number of last layers wrapped with LoRA")
    parser.add_argument("--condition", choices=sorted(CONDITIONS), help="A/B/C/D experiment condition")
    parser.add_argument("--context-file", help="Optional source context file for B/D prompts")
    parser.add_argument("--suggestions-file", help="Optional question suggestions file: txt, JSONL, or JSON array")
    parser.add_argument("--reference", default="", help="Optional source reference line for B/D prompts")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens per turn")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--no-stream", action="store_true", help="Print after full generation instead of streaming")
    parser.add_argument("--stop-strings", nargs="*", default=DEFAULT_STOP_STRINGS, help="Stop generation strings")
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


def read_suggestions(suggestions_file: str | None) -> list[str]:
    if not suggestions_file:
        return DEFAULT_SUGGESTIONS

    content = Path(suggestions_file).read_text().strip()
    if not content:
        return DEFAULT_SUGGESTIONS

    suggestions = []
    if content.startswith("["):
        values = json.loads(content)
        for value in values:
            if isinstance(value, str):
                suggestions.append(value.strip())
            elif isinstance(value, dict) and isinstance(value.get("question"), str):
                suggestions.append(value["question"].strip())
            elif isinstance(value, dict) and isinstance(value.get("text"), str):
                question = extract_question(value["text"])
                if question:
                    suggestions.append(question)
    else:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                if isinstance(row.get("question"), str):
                    suggestions.append(row["question"].strip())
                elif isinstance(row.get("text"), str):
                    question = extract_question(row["text"])
                    if question:
                        suggestions.append(question)
            else:
                suggestions.append(line)

    return list(dict.fromkeys(s for s in suggestions if s)) or DEFAULT_SUGGESTIONS


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


def print_suggestions(suggestions: list[str], limit: int = 10) -> None:
    print("Suggested questions:")
    for index, question in enumerate(suggestions[:limit], start=1):
        print(f"  {index}. {question}")
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


def print_help() -> None:
    print("Commands:")
    print("  /help       tampilkan bantuan")
    print("  /context    paste dokumen sumber untuk mode B/D")
    print("  /suggest    tampilkan contoh pertanyaan")
    print("  /clear      hapus dokumen sumber")
    print("  /show       tampilkan status model/mode/context")
    print("  /exit       keluar")
    print("Ketik pertanyaan langsung untuk mendapat jawaban.")


def main() -> None:
    args = build_parser().parse_args()
    preset = resolve_preset(args)
    condition = resolve_condition(args, preset)

    adapter_file = preset.adapter_file if needs_adapter(condition) else None
    if needs_adapter(condition) and not adapter_file:
        raise ValueError(f"Mode {condition} membutuhkan adapter, tetapi preset ini tidak punya adapter_file.")

    print(f"Loading model: {preset.label}")
    print(f"Mode: {condition} - {CONDITIONS[condition]}")
    model, tokenizer = prepare_model(preset.model, adapter_file, preset.lora_layers)
    stop_token_sequences = build_stop_token_sequences(tokenizer, args.stop_strings)
    context = read_context(args.context_file)
    suggestions = read_suggestions(args.suggestions_file)
    reference = args.reference.strip()

    if needs_context(condition) and not context:
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
            print(f"Model: {preset.label}")
            print(f"Mode: {condition} - {CONDITIONS[condition]}")
            print(f"Adapter: {adapter_file or '-'}")
            print(f"Context chars: {len(context)}")
            print(f"Suggestions: {len(suggestions)}")
            continue

        if user_input.isdigit():
            suggestion_index = int(user_input) - 1
            if 0 <= suggestion_index < len(suggestions):
                user_input = suggestions[suggestion_index]
                print(f"Q: {user_input}")
            else:
                print("Nomor suggestion tidak tersedia. Pakai /suggest untuk melihat daftar.")
                continue

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
