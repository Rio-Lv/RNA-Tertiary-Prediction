import os
from Sequence import SequenceDataset, Sequence
from SpineModel import SpineModel
from tools import *

# set here to cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    spine_model = SpineModel()
    seq_dataset = SequenceDataset(n_sequences=10, sequence_size=5)

    test_seq = seq_dataset.source_sequences[7]
    test_seq_str = test_seq.seq_str

    source_coords = test_seq.coords
    generated_coords, recording = spine_model.construct_spine_coords(
        seq_str=test_seq_str, n_iter=300, max_delta=0.02
    )
    target_pdb_path = "seq_output/spine_test_source.pdb"
    generated_pdb_path = "seq_output/spine_test_generated.pdb"
    Sequence.to_pdb(
        coords=source_coords, seq_str=test_seq_str, save_path=target_pdb_path
    )
    Sequence.to_pdb(
        coords=generated_coords, seq_str=test_seq_str, save_path=generated_pdb_path
    )
    Sequence.compute_similarity_us_align(
        gen_path=generated_pdb_path, target_path=target_pdb_path
    )
    
    create_video(source_coords,recording, 5, "seq_output/spine_model_test.mp4")

    plot_coords_list([source_coords, generated_coords], ["source", "generated"])


# TODO: Maybe what needs to be learnt is the deltas.. Im thinking in windows, 
# Input is distance matrix (windowed) + (encoding) => output is (deltas) then use whole simulation as training..
# Would Spine Model even be needed then? Only if it helps correct the Mirror / Symmetry issues