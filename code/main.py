import numpy as np
from gcn import GCNTrainer
from arg_parser import parameter_parser
from utils import tab_printer, read_graph, best_printer, setup_features


def main():
    args = parameter_parser()
    tab_printer(args)
    edges = read_graph(args)  # number of edges --> otc: 35592, alpha: 24186
    setup_features(args)

    best = [["# Training Timeslots", "Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro", "Run Time"]]

    times = 8 if args.single_prediction else 6
    # single_prediction: {1-2}-->3, {1-3}-->4, {1-4}-->5, {1-5}-->6, {1-6}-->7, {1-7}-->8, {1-8}-->9, {1-9}-->10
    # multi_prediction: {1-2}-->{3-5}, {1-3}-->{4-6}, {1-4}-->{5-7}, {1-5}-->{6-8}, {1-6}-->{7-9}, {1-7}-->{8-10}
    for t in range(times):
        trainer = GCNTrainer(args, edges)
        trainer.setup_dataset()

        print("Ready, Go! Round = " + str(t))
        trainer.create_and_train_model()

        best_epoch = [0, 0, 0, 0, 0, 0, 0]
        for i in trainer.logs["performance"][1:]:
            # sum of MCC, AUC, ACC_Balanced, F1_Macro
            if float(i[1]+i[2]+i[3]+i[6]) > (best_epoch[1]+best_epoch[2]+best_epoch[3]+best_epoch[6]):
                best_epoch = i

        best_epoch.append(trainer.logs["training_time"][-1][1])
        best_epoch.insert(0, t + 2)
        best.append(best_epoch)

        args.train_time_slots += 1

    print("\nBest results of each run")
    best_printer(best)

    print("\nMean, Max, Min, Std")
    analyze = np.array(best)[1:, 1:].astype(np.float64)
    mean = np.mean(analyze, axis=0)
    maxi = np.amax(analyze, axis=0)
    mini = np.amin(analyze, axis=0)
    std = np.std(analyze, axis=0)
    results = [["Epoch", 'MCC', "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro", "Run Time"], mean, maxi, mini, std]

    best_printer(results)


if __name__ == "__main__":
    main()
