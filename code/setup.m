%%
clear;
clc;

%% add paths
addpath('../data/');
addpath('./tensorlab/');
addpath(genpath('./toolbox_graph/'));

%% load data
load('core_tensor_truncated_id_50_exp_25.mat');
load('faces.mat');
load('features.mat');
load('corr_backup.mat');
load('used_img_idx.mat');
load('U_exp_truncated_25.mat');
load('img_all.mat');

%% load all pictures
fprintf('loading pictures...\n');
used_img_idx = used_img_idx(1:20);
img_all = img_all(used_img_idx);

%% load features
fprintf('loading features...\n');
features = zeros(size(features_all, 1), length(used_img_idx)*2);
for i = 1:length(used_img_idx)
    features(:, i*2-1:i*2) = features_all(:, used_img_idx(i)*2-1:used_img_idx(i)*2);
end

%%
fprintf('face fitting...\n');
corr_current = [corr_eyebrow; corr_eye; corr_nose; corr_mouth; corr_contour];
W = face_mesh_fitting(core_tensor_truncated, faces, corr_current, features, 3, img_all);

%%
fprintf('display blendshape...\n');
EXP = cal_blendshapes(core_tensor_truncated, faces, U_exp_truncated, W(4).second_step.X, 1);