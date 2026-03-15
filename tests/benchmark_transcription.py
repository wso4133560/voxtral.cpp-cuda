#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass(frozen=True)
class Fixture:
    name: str
    audio_path: Path
    expected_text: str
    duration_s: float


@dataclass(frozen=True)
class RunResult:
    sample: str
    backend: str
    threads: int
    duration_s: float
    wall_time_s: float
    generated_tokens: int
    tokens_per_s: float
    real_time_factor: float
    wer: float
    accuracy: float
    transcript: str
    expected_text: str


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    dist = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        dist[i][0] = i
    for j in range(len(hyp_words) + 1):
        dist[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dist[i][j] = dist[i - 1][j - 1]
            else:
                dist[i][j] = min(
                    dist[i - 1][j],
                    dist[i][j - 1],
                    dist[i - 1][j - 1],
                ) + 1

    if not ref_words:
        return 0.0
    return float(dist[len(ref_words)][len(hyp_words)]) / float(len(ref_words))


def normalize_text(text: str) -> str:
    upper = text.upper()
    upper = re.sub(r"[^A-Z0-9'\s]+", " ", upper)
    upper = re.sub(r"\s+", " ", upper)
    return upper.strip()


def audio_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    return frames / float(sample_rate)


def discover_fixtures(samples_dir: Path) -> list[Fixture]:
    fixtures: list[Fixture] = []
    for wav_path in sorted(samples_dir.glob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        fixtures.append(
            Fixture(
                name=wav_path.stem,
                audio_path=wav_path,
                expected_text=txt_path.read_text(encoding="utf-8").strip(),
                duration_s=audio_duration_s(wav_path),
            )
        )
    return fixtures


def run_fixture(
    voxtral_bin: Path,
    model_path: Path,
    fixture: Fixture,
    threads: int,
    max_tokens: int,
    use_cuda: bool,
) -> RunResult:
    with tempfile.TemporaryDirectory(prefix=f"voxtral_bench_{fixture.name}_") as td:
        temp_dir = Path(td)
        transcript_path = temp_dir / "transcript.txt"
        tokens_path = temp_dir / "tokens.txt"

        cmd = [
            str(voxtral_bin),
            "--model",
            str(model_path),
            "--audio",
            str(fixture.audio_path),
            "--threads",
            str(threads),
            "--n-tokens",
            str(max_tokens),
            "--output-text",
            str(transcript_path),
            "--dump-tokens",
            str(tokens_path),
            "--log-level",
            "warn",
        ]
        if use_cuda:
            cmd.append("--cuda")

        started_at = time.perf_counter()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        wall_time_s = time.perf_counter() - started_at

        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise RuntimeError(f"inference failed for sample {fixture.name}")

        transcript = transcript_path.read_text(encoding="utf-8").strip()
        raw_tokens = tokens_path.read_text(encoding="utf-8").strip()
        token_ids = [int(item) for item in raw_tokens.split()] if raw_tokens else []
        token_count = len(token_ids)
        wer = compute_wer(fixture.expected_text, transcript)
        tokens_per_s = token_count / wall_time_s if wall_time_s > 0 else 0.0
        real_time_factor = wall_time_s / fixture.duration_s if fixture.duration_s > 0 else 0.0

        return RunResult(
            sample=fixture.name,
            backend="cuda" if use_cuda else "cpu",
            threads=threads,
            duration_s=fixture.duration_s,
            wall_time_s=wall_time_s,
            generated_tokens=token_count,
            tokens_per_s=tokens_per_s,
            real_time_factor=real_time_factor,
            wer=wer,
            accuracy=1.0 - wer,
            transcript=transcript,
            expected_text=fixture.expected_text,
        )


def print_results(results: list[RunResult]) -> None:
    print(
        f"{'Sample':<18} {'Backend':<6} {'Audio(s)':>8} {'Wall(s)':>8} "
        f"{'Tokens':>8} {'Tok/s':>9} {'RTF':>8} {'WER':>8} {'Acc':>8}"
    )
    print("-" * 87)
    for result in results:
        print(
            f"{result.sample:<18} {result.backend:<6} {result.duration_s:>8.2f} "
            f"{result.wall_time_s:>8.2f} {result.generated_tokens:>8d} "
            f"{result.tokens_per_s:>9.2f} {result.real_time_factor:>8.2f} "
            f"{result.wer:>8.4f} {result.accuracy:>7.2%}"
        )

    avg_accuracy = sum(item.accuracy for item in results) / len(results)
    avg_wer = sum(item.wer for item in results) / len(results)
    total_tokens = sum(item.generated_tokens for item in results)
    total_wall_time = sum(item.wall_time_s for item in results)
    total_audio = sum(item.duration_s for item in results)
    aggregate_tok_s = total_tokens / total_wall_time if total_wall_time > 0 else 0.0
    aggregate_rtf = total_wall_time / total_audio if total_audio > 0 else 0.0

    print("-" * 87)
    print(
        f"{'AVERAGE':<18} {results[0].backend:<6} {total_audio / len(results):>8.2f} "
        f"{total_wall_time / len(results):>8.2f} {total_tokens // len(results):>8d} "
        f"{aggregate_tok_s:>9.2f} {aggregate_rtf:>8.2f} {avg_wer:>8.4f} {avg_accuracy:>7.2%}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Voxtral transcription accuracy and token speed.")
    parser.add_argument("--voxtral-bin", default="build/voxtral", help="Path to voxtral binary.")
    parser.add_argument("--model", default="models/voxtral/Q4_0.gguf", help="Path to GGUF model.")
    parser.add_argument("--samples-dir", default="samples", help="Directory containing wav/txt fixture pairs.")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads to pass into voxtral.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum decode tokens.")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA backend.")
    parser.add_argument("--json", action="store_true", help="Emit JSON in addition to the table.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    voxtral_bin = Path(args.voxtral_bin)
    model_path = Path(args.model)
    samples_dir = Path(args.samples_dir)

    missing = [str(path) for path in [voxtral_bin, model_path, samples_dir] if not path.exists()]
    if missing:
        for item in missing:
            print(f"missing required path: {item}", file=sys.stderr)
        return 1

    fixtures = discover_fixtures(samples_dir)
    if not fixtures:
        print("no wav/txt fixtures found", file=sys.stderr)
        return 1

    results = [
        run_fixture(
            voxtral_bin=voxtral_bin,
            model_path=model_path,
            fixture=fixture,
            threads=args.threads,
            max_tokens=args.max_tokens,
            use_cuda=args.cuda,
        )
        for fixture in fixtures
    ]

    print_results(results)
    if args.json:
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
