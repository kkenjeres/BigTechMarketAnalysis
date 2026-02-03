import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from data_utils import load_raw


def save_fig1():
    df = load_raw()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title('Рисунок 1 – Исходный набор данных (Объединенный: AMZN, GOOGL, META, MSFT)', fontsize=12, fontweight='bold', y=1.05)

    sample = df.groupby('Company').head(2)
    table = ax.table(cellText=sample.values, colLabels=sample.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.3)

    plt.savefig('images/Figure_1_InitialData.png', bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == "__main__":
    save_fig1()
