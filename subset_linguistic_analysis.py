from tqdm.auto import tqdm 
import pickle
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="subset linguistic analysis")
    parser.add_argument(
        "--partition_indices",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()
    freq_dict={}
    with open("bert_first_sentences.json", "r") as f:
        all_lines=f.readlines()
    print("loaded bert_first_sentences")
    with open(args.partition_indices, "rb") as f:
        partition_indices=pickle.load(f)
    print("loaded partitions")
    pbar=tqdm(range(len(partition_indices)))
    for partition in partition_indices:
        ind=partition[:int(0.25*len(partition))]
        for i in ind:
            for id in all_lines[i][14:-3].split(","):
                if id in freq_dict:
                    freq_dict[id]+=1
                else:
                    freq_dict[id]=1
        pbar.update(1)
    with open(args.output_file, "wb") as f:
        pickle.dump(freq_dict, f)
    
if __name__=="__main__":
    main()