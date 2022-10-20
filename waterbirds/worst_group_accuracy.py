import numpy as np
import matplotlib.pyplot as plt
from arguments import args

# Hyperparameters
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
MOMENTUM = args.momentum

BASE_LRS = [1e-03, 1e-04, 1e-05]
WEIGHT_DECAYS = [1e-04, 1e-01, 1.0]
RUNS = [3]
STAGE_1_EPOCHS = [25, 40, 50, 60, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
UPSAMPLES = [20, 50, 100]
valid_avg_accuracy_threshold_lower = 94
valid_avg_accuracy_threshold_upper = valid_avg_accuracy_threshold_lower + 0.5
worst_group_accuracy_list = []

for BASE_LR, WEIGHT_DECAY in zip(BASE_LRS, WEIGHT_DECAYS):
    for RUN in RUNS:
        for STAGE_1_EPOCH in STAGE_1_EPOCHS:
            for UPSAMPLE in UPSAMPLES: 
                STAGE_2_PATH = 'saved_models/run_%s/bs_%s_epochs_%s_lr_%s_wd_%s/epoch_%s/stage_2/incorrect_upsampled_%s_bs_%s_epochs_%s_lr_%s_wd_%s/'%(RUN, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY, STAGE_1_EPOCH, UPSAMPLE, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY)
                
                valid_avg_acc = np.loadtxt(STAGE_2_PATH + "plots/valid_avg_acc.txt")
                valid_waterbirds_waterbkgd_acc = np.loadtxt(STAGE_2_PATH + "plots/valid_waterbirds_waterbkgd_acc.txt")
                valid_waterbirds_landbkgd_acc = np.loadtxt(STAGE_2_PATH + "plots/valid_waterbirds_landbkgd_acc.txt")
                valid_landbirds_waterbkgd_acc = np.loadtxt(STAGE_2_PATH + "plots/valid_landbirds_waterbkgd_acc.txt")
                valid_landbirds_landbkgd_acc = np.loadtxt(STAGE_2_PATH + "plots/valid_landbirds_landbkgd_acc.txt")
                valid_worst_group_acc = [min(a, b, c, d) for a,b,c,d in zip(valid_waterbirds_waterbkgd_acc, valid_waterbirds_landbkgd_acc, valid_landbirds_waterbkgd_acc, valid_landbirds_landbkgd_acc)]

                test_avg_acc = np.loadtxt(STAGE_2_PATH + "plots/test_avg_acc.txt")
                test_waterbirds_waterbkgd_acc = np.loadtxt(STAGE_2_PATH + "plots/test_waterbirds_waterbkgd_acc.txt")
                test_waterbirds_landbkgd_acc = np.loadtxt(STAGE_2_PATH + "plots/test_waterbirds_landbkgd_acc.txt")
                test_landbirds_waterbkgd_acc = np.loadtxt(STAGE_2_PATH + "plots/test_landbirds_waterbkgd_acc.txt")
                test_landbirds_landbkgd_acc = np.loadtxt(STAGE_2_PATH + "plots/test_landbirds_landbkgd_acc.txt")
                test_worst_group_acc = [min(a, b, c, d) for a,b,c,d in zip(test_waterbirds_waterbkgd_acc, test_waterbirds_landbkgd_acc, test_landbirds_waterbkgd_acc, test_landbirds_landbkgd_acc)]

                correct_accuracies_waterbirds = np.loadtxt(STAGE_2_PATH + "plots/correct_accuracies_waterbirds.txt")
                incorrect_accuracies_waterbirds = np.loadtxt(STAGE_2_PATH + "plots/incorrect_accuracies_waterbirds.txt")
                estimated_worst_group_acc_waterbirds = [min(x, y) for x,y in zip(correct_accuracies_waterbirds, incorrect_accuracies_waterbirds)]
                correct_accuracies_landbirds = np.loadtxt(STAGE_2_PATH + "plots/correct_accuracies_landbirds.txt")
                incorrect_accuracies_landbirds = np.loadtxt(STAGE_2_PATH + "plots/incorrect_accuracies_landbirds.txt")
                estimated_worst_group_acc_landbirds = [min(x, y) for x,y in zip(correct_accuracies_landbirds, incorrect_accuracies_landbirds)]
                estimated_worst_group_acc = [min(a, b, c, d) for a,b,c,d in zip(correct_accuracies_waterbirds, incorrect_accuracies_waterbirds, correct_accuracies_landbirds, incorrect_accuracies_landbirds)]
                
                worst_group_accuracy = list(zip(list(range(1, 301)), 
                                             estimated_worst_group_acc_waterbirds, #incorrect_accuracies_waterbirds, 
                                             estimated_worst_group_acc_landbirds, #incorrect_accuracies_landbirds,
                                             estimated_worst_group_acc,
                                             valid_avg_acc, 
                                             valid_worst_group_acc,
                                             test_avg_acc, 
                                             test_worst_group_acc,
                                             [STAGE_1_EPOCH]*len(list(range(1, 301))),
                                             [UPSAMPLE]*len(list(range(1, 301))),
                                             [BASE_LR]*len(list(range(1, 301))),
                                             [WEIGHT_DECAY]*len(list(range(1, 301)))))
                worst_group_accuracy = [a for a in worst_group_accuracy if a[4] >= valid_avg_accuracy_threshold_lower and a[4] < valid_avg_accuracy_threshold_upper]
                worst_group_accuracy_list += worst_group_accuracy

#epochs = [a[0] for a in worst_group_accuracy_list]
#incorrect_accuracies_waterbirds = [a[1] for a in worst_group_accuracy_list]
#incorrect_accuracies_landbirds = [a[2] for a in worst_group_accuracy_list]
#estimated_worst_group_acc = [a[3] for a in worst_group_accuracy_list]
#valid_avg_acc = [a[4] for a in worst_group_accuracy_list]
#valid_worst_group_acc = [a[5] for a in worst_group_accuracy_list]
#test_avg_acc = [a[6] for a in worst_group_accuracy_list]
#test_worst_group_acc = [a[7] for a in worst_group_accuracy_list]

#plt.figure()
#plt.scatter(incorrect_accuracies_waterbirds, incorrect_accuracies_landbirds)
#plt.xlabel('Waterbirds Incorrect Accuracy')
#plt.ylabel('Landbirds Incorrect Accuracy')
#plt.grid()
#for i in range(len(epochs)):
#    plt.annotate((round(test_avg_acc[i],1), round(test_worst_group_acc[i],1)), (incorrect_accuracies_waterbirds[i], incorrect_accuracies_landbirds[i]), fontsize=6)
#plt.savefig('./worst_group_accuracy_pareto_points.png')
#plt.close()

print('>=', valid_avg_accuracy_threshold_lower, 'and <', valid_avg_accuracy_threshold_upper)

estimated_worst_group_accuracy_list_sorted = sorted(worst_group_accuracy_list, key=lambda ele : ele[3])
estimated_worst_group_accuracy = estimated_worst_group_accuracy_list_sorted[-1]
estimated_test_avg_acc = estimated_worst_group_accuracy[6]
estimated_test_worst_group_accuracy = estimated_worst_group_accuracy[7]
print('Estimated Test Average Accuracy: ', estimated_test_avg_acc)
print('Estimated Worst-group Accuracy: ', estimated_test_worst_group_accuracy)
#print(estimated_worst_group_accuracy[4], estimated_worst_group_accuracy[5])

ground_truth_worst_group_accuracy_list_sorted = sorted(worst_group_accuracy_list, key=lambda ele : ele[5])
ground_truth_worst_group_accuracy = ground_truth_worst_group_accuracy_list_sorted[-1]
ground_truth_test_avg_acc = ground_truth_worst_group_accuracy[6]
ground_truth_test_worst_group_accuracy = ground_truth_worst_group_accuracy[7]
print('Ground Truth Test Average Accuracy: ', ground_truth_test_avg_acc)
print('Ground Truth Worst-group Accuracy: ', ground_truth_test_worst_group_accuracy)
#print(ground_truth_worst_group_accuracy[4], ground_truth_worst_group_accuracy[5])

