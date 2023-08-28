
# Load vectors
# vectors = np.load('api/img_vectors.npy')

# Load ids
with open('id_to_label.txt', 'r') as f:
    ids = [line.strip().split('\t')[0] for line in f]

print(ids) 