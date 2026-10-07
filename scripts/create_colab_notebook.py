from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def code_cell(source: str):
    return nbf.v4.new_code_cell(source.strip())


def md_cell(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md_cell(
            "# Wi-Fi/Cellular UAV Mobility Prediction and Trajectory Optimization\n"
            "Full Colab execution notebook. Runtime -> Change runtime type -> GPU before running."
        ),
        md_cell(
            "## Phase 0: Get the code\n"
            "Clones the repo and installs it as a package so every `!python scripts/...` cell can import "
            "`wifi_uav`. Set `PRESET` to `smoke` for a 2-minute check, `medium` for a quick result, or "
            "`full` for the final results (GPU recommended)."
        ),
        code_cell(
            "import os\n"
            "REPO_URL = 'https://github.com/sahilsingh069/wifi-uav-project.git'\n"
            "PROJECT_ROOT = '/content/wifi-uav-project'\n"
            "PRESET = 'full'  # 'smoke' | 'medium' | 'full'\n"
            "if not os.path.isdir(PROJECT_ROOT):\n"
            "    !git clone -q {REPO_URL} {PROJECT_ROOT}\n"
            "os.chdir(PROJECT_ROOT)\n"
            "!pip install -q -e .\n"
            "print(os.getcwd(), PRESET)"
        ),
        code_cell(
            "import torch\n"
            "print('CUDA available:', torch.cuda.is_available())\n"
            "print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
        ),
        md_cell("## Phase 1: RF dataset generation (3GPP UMi/UMa channel, correlated shadowing)"),
        code_cell("!python scripts/phase1_collect_data.py --preset {PRESET}"),
        md_cell("## Phase 2: Feature extraction (split by episode to avoid leakage)"),
        code_cell("!python scripts/phase2_build_features.py --preset {PRESET}"),
        md_cell("## Phase 3: LSTM-Transformer training"),
        code_cell("!python scripts/phase3_train_predictor.py --preset {PRESET}"),
        md_cell("## Phase 4: UAV environment sanity check (UAVs observe predicted user positions)"),
        code_cell("!python scripts/phase4_sanity_env.py --preset {PRESET}"),
        md_cell("## Phase 5: MADDPG training"),
        code_cell("!python scripts/phase5_train_maddpg.py --preset {PRESET}"),
        md_cell("## Phase 6: Evaluation against baselines"),
        code_cell("!python scripts/phase6_evaluate.py --preset {PRESET}"),
        code_cell(
            "from IPython.display import Image, display\n"
            "for name in ['training_curves', 'baseline_comparison', 'coverage_heatmap']:\n"
            "    path = f'results/{name}.png'\n"
            "    if os.path.exists(path):\n"
            "        display(Image(path))"
        ),
        code_cell("!zip -qr wifi_uav_results.zip data/normalization.npz checkpoints results"),
    ]
    out = Path("notebooks/WiFi_UAV_Full_Project_Colab.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
