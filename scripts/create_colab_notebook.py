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
            "Full Colab execution notebook."
        ),
        code_cell("!pip install -q 'numpy<2' pandas scikit-learn torch matplotlib seaborn tqdm gymnasium nbformat"),
        code_cell(
            "import torch\n"
            "print('CUDA available:', torch.cuda.is_available())\n"
            "print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
        ),
        md_cell(
            "## Phase 0: Clone or upload project\n"
            "Upload this repository folder to Colab or mount Google Drive before running the phase scripts."
        ),
        code_cell(
            "import os, sys\n"
            "PROJECT_ROOT = '/content/wifi_uav_project'\n"
            "if os.path.isdir(PROJECT_ROOT):\n"
            "    os.chdir(PROJECT_ROOT)\n"
            "    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))\n"
            "print(os.getcwd())"
        ),
        md_cell("## Phase 1: RF dataset generation"),
        code_cell("!python scripts/phase1_collect_data.py --preset smoke"),
        md_cell("## Phase 2: Feature extraction"),
        code_cell("!python scripts/phase2_build_features.py --preset smoke"),
        md_cell("## Phase 3: LSTM-Transformer training"),
        code_cell("!python scripts/phase3_train_predictor.py --preset smoke --epochs 2"),
        md_cell("## Phase 4: UAV environment sanity check"),
        code_cell("!python scripts/phase4_sanity_env.py --preset smoke"),
        md_cell("## Phase 5: MADDPG training"),
        code_cell("!python scripts/phase5_train_maddpg.py --preset smoke"),
        md_cell("## Phase 6: Evaluation"),
        code_cell("!python scripts/phase6_evaluate.py --preset smoke"),
        code_cell("!zip -r wifi_uav_results.zip data models checkpoints results reports notebooks src scripts README.md pyproject.toml"),
    ]
    out = Path("notebooks/WiFi_UAV_Full_Project_Colab.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
