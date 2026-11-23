# Summary of all approaches (test set, evaluated once per selected model)

| Approach | Test acc [95% CI] | Test macro-F1 [95% CI] | Train acc | Val acc | Train−test gap | Trainable params | Training time (s) | Selected |
|---|---|---|---|---|---|---|---|---|
| full_finetune | 0.8490 [0.8384, 0.8605] | 0.8464 [0.8349, 0.8572] | 0.9990 | 0.8954 | 0.1500 | 1,555,781 | 251.9 | epoch 11 of 15 |
| head_only | 0.8673 [0.8566, 0.8779] | 0.8658 [0.8545, 0.8755] | 0.9915 | 0.8927 | 0.1242 | 628,773 | 83.8 | epoch 10 of 15 |
| features_logreg | 0.8569 [0.8460, 0.8681] | 0.8551 [0.8437, 0.8655] | 0.9881 | 0.8764 | 0.1312 | 21,349 | 93.1 (extract 93.0 + fit 0.1) | C=0.01 |
| features_linear_svm | 0.8335 [0.8217, 0.8457] | 0.8309 [0.8182, 0.8419] | 0.9990 | 0.8750 | 0.1655 | 21,349 | 99.6 (extract 93.0 + fit 6.6) | C=0.01 |
| features_random_forest | 0.8277 [0.8163, 0.8397] | 0.8232 [0.8112, 0.8344] | 1.0000 | 0.8614 | 0.1723 | n/a (forest) | 96.5 (extract 93.0 + fit 3.5) | max_depth=None |

Train acc = selected model on training images without augmentation. CIs: bootstrap over test images, 2000 resamples.
