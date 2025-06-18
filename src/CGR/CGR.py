import matplotlib.pyplot as plt


def read_fasta(file_path):
    sequence = ""
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('>'):
                sequence += line.strip()
    return sequence

def chaos_game_representation(sequence):
    # Mapping nucleotides to 2D positions
    mapping = {
        'A': (0, 0),
        'T': (0, 1),
        'G': (1, 0),
        'C': (1, 1)
    }

    # Initialize starting point
    x, y = 0.5, 0.5
    points_x, points_y, colors = [], [], []

    # Color mapping
    color_map = {
        'A': 'red',
        'T': 'blue',
        'G': 'green',
        'C': 'orange'
    }

    # Iterate through the sequence
    for nucleotide in sequence:
        if nucleotide in mapping:
            dx, dy = mapping[nucleotide]
            x = (x + dx) / 2
            y = (y + dy) / 2
            points_x.append(x)
            points_y.append(y)
            colors.append(color_map[nucleotide])

    return points_x, points_y, colors

def plot_cgr(points_x, points_y, colors):
    plt.figure(figsize=(8, 8))
    plt.scatter(points_x, points_y, c=colors, s=1, alpha=0.7)
    plt.title('Chaos Game Representation (CGR)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def main():
    # Path to your FASTA file
    fasta_file = 'US1.fasta'
    sequence = read_fasta(fasta_file)
    points_x, points_y, colors = chaos_game_representation(sequence)
    plot_cgr(points_x, points_y, colors)

if __name__ == "__main__":
    main()