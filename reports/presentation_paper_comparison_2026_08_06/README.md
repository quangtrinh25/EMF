# Paper-comparison presentation

- [Slide deck PDF](SLIDE_DECK.pdf)
- [Detailed Vietnamese speaker script](SLIDE_SCRIPT.md)
- [Slide outline CSV](SLIDE_OUTLINE.csv)
- Individual 16:9 PNG slides: `slides/`

Rebuild:

```bash
# Run from the cloned repository root.
./.venv/bin/python reporting/build_paper_comparison_presentation.py \
  --out_dir reports/presentation_paper_comparison_2026_08_06
```

This is reporting-only. It does not read final labels/predictions, train, infer, or change deployment.
