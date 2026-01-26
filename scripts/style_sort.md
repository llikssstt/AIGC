# Chinese Landscape Painting Style Sorter

`style_sort.py` (inside `scripts/`) reorganizes every image under `Chinese-Landscape-Painting-Dataset`
into the three style folders **水墨**, **工笔**, and **青绿** by:

1. Extracting a few quick color/brightness statistics from each image.
2. Asking the local Qwen3-1.7B model which style best matches those statistics.
3. Copying the file into `sorted_by_style/<style>` with a normalized prefix (`shimo_`, `gongbi_`,
   or `qinglv_`) and a zero-padded counter.

## Requirements

- Python 3.10+
- The dataset in `Chinese-Landscape-Painting-Dataset` (already present in this repo).
- The Qwen3-1.7B checkpoint unpacked at `E:\AIGC\models\Qwen3-1.7B` (or point `--model` to your own copy).
- Packages: `pip install torch transformers pillow numpy`.

If you are using a GPU, the script will automatically choose `cuda`; pass `--device cpu` if you prefer
CPU-only inference.

## Running

From the repo root, run:

```bash
python scripts/style_sort.py \
  --model "E:\AIGC\models\Qwen3-1.7B" \
  --dataset "Chinese-Landscape-Painting-Dataset" \
  --output "Chinese-Landscape-Painting-Dataset/sorted_by_style"
```

Add `--dry-run` to preview moves, or `--device cpu` to avoid CUDA. The script logs progress and
keeps counts for each folder; when it finishes you will have three new folders under the output path.

## Accuracy improvements

- The sorter now reads ink/style definitions (墨韵、工笔线条、青绿矿物色) and feeds them into every
  Qwen prompt, which reduces hallucination and sharpens the difference between the three genres.
- Built-in heuristics detect starkly monochrome works as 水墨 and strongly green/blue, high-saturation works
  as 青绿 or 工笔 so the LLM only runs when the image sits in the gray zone. This slows the sorting down
  a bit but improves consistency.
- Keep running `--dry-run` or inspect the logs (`INFO` level) if you suspect a misclassification; they now report
  whether the decision came from a heuristic or from Qwen’s answer.

## Notes

- The sorter does not delete originals; it copies them so the raw dataset remains untouched.
- Classification is best-effort—if Qwen3 fails to mention a style name verbatim the script falls back to
  a simple heuristic.
