import matplotlib.pyplot as plt
from ubongo import PIECES, generate_puzzle_with_mandatory_alt


def main():
    ratios = []
    for seed in range(100):
        puzzle = generate_puzzle_with_mandatory_alt(
            library=PIECES,
            k=3,
            max_w=8,
            max_h=6,
            mandatory_piece="I3",
            seed=seed,
            max_attempts=600,
        )
        if puzzle is not None:
            ratios.append(puzzle["circumference_ratio"])
        else:
            print(f"No puzzle found for seed {seed}; skipping.")

    plt.hist(ratios, bins="auto", edgecolor="black")
    plt.title("Circumference Ratio Distribution for 100 Ubongo Puzzles")
    plt.xlabel("Circumference / Area ratio")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
