%for i = 1:20
    %[X,FVAL,EXITFLAG,OUTPUT] = fmincon(@(x)test_optimization_simple(x, ia, core_tensor, U, features, 9), X0, [], [], [], [], [], [], [], options);
    %[X,FVAL,EXITFLAG,OUTPUT] = fminunc(@(x)test_optimization_simple(x, ia, core_tensor_t, U_t, features, 20), X, options_unc);
    %X_60{i} = X; FVAL_60{i} = FVAL; EXITFLAG_60{i} = EXITFLAG; OUTPUT_60{i} = OUTPUT; save(['z' num2str(i) '.mat'], 'X', 'FVAL', 'EXITFLAG', 'OUTPUT');
%end

%%
w_id = zeros(50, 1);
w_id(1) = 1;

w_exp = zeros(25, 1);
w_exp(1) = 1;
% 
% w_id = X_result(7:56);
% w_exp = X_result(57:81);

% face_reconstructed = tmprod(core_tensor_truncated, {w_id, U_exp_truncated'*w_exp}, [2, 3], 'T');
face_reconstructed = tmprod(core_tensor_truncated, {w_id, w_exp}, [2, 3], 'T');
face_reconstructed = reshape(face_reconstructed, 3, 11510)';
figure; plot_mesh(face_reconstructed, faces); camlight('headlight');

%%
file_name = '../data/FaceWarehouse_Data_0/Tester_57/Blendshape/shape_36.obj';
fid = fopen(file_name);
frewind(fid);
face_openmouth = zeros(11510, 3);
for i = 1:11510
    l = fgetl(fid);
    face_openmouth(i, :) = sscanf(l(3:end), '%f %f %f');
end
fclose(fid);
figure; plot_mesh(face_openmouth, faces);
camlight('headlight');
%% get landmark index
left_eye_pos = zeros(length(left_eye), 3);
for i = 1:length(left_eye)
    left_eye_pos(i, :) = left_eye(i).Position;
end
[~, left_eye_idx] = ismember(left_eye_pos, face_reconstructed, 'rows');

right_eye_pos = zeros(length(right_eye), 3);
for i = 1:length(right_eye)
    right_eye_pos(i, :) = right_eye(i).Position;
end
[~, right_eye_idx] = ismember(right_eye_pos, face_reconstructed, 'rows');

%%
mouth_pos = zeros(length(mouth), 3);
for i = 1:length(mouth)
    mouth_pos(i, :) = mouth(i).Position;
end
[~, mouth_idx] = ismember(mouth_pos, face_reconstructed, 'rows');

%%
mouth_idx = zeros(length(mouth_datatip), 1);
for i = 1:length(mouth_datatip)
    mouth_idx(i) = mouth_datatip(i).DataIndex;
end


%%
nose_pos = zeros(length(nose), 3);
for i = 1:length(nose)
    nose_pos(i, :) = nose(i).Position;
end
[~, nose_idx] = ismember(nose_pos, face_reconstructed, 'rows');

eye_brow_pos = zeros(length(eye_brow), 3);
for i = 1:length(eye_brow)
    eye_brow_pos(i, :) = eye_brow(i).Position;
end
[~, eye_brow_idx] = ismember(eye_brow_pos, face_reconstructed, 'rows');

%%
contour_pos = zeros(length(contour_p), 3);
for i = 1:length(contour_p)
    contour_pos(i, :) = contour_p(i).Position;
end
% [~, contour_idx] = ismember(contour_p, face_openmouth, 'rows');
contour_idx = knnsearch(face_openmouth, contour_pos);

%% display landmarks on 3D mesh
figure;
% plot_mesh(face_reconstructed, faces); camlight('headlight');
% hold on; 
plot3(face_reconstructed(:, 1), face_reconstructed(:, 2), face_reconstructed(:, 3), '.', 'MarkerSize', 10);
hold on;
% figure; plot3(face_reconstructed(idx, 1), face_reconstructed(idx, 2), face_reconstructed(idx, 3), '.', 'MarkerSize', 10);
for i = 1:size(idx, 1)
    hold on;
    plot3(face_reconstructed(idx(i), 1), face_reconstructed(idx(i), 2), face_reconstructed(idx(i), 3), ...
        'r.', 'MarkerSize', 10);
    hold on;
    text(face_reconstructed(idx(i), 1), face_reconstructed(idx(i), 2), face_reconstructed(idx(i), 3), ...
        sprintf('\\leftarrow %d-%d', i, idx(i)));
end
axis off

%% display corr. feature points on pic
figure; imshow(img_all{1}); hold on;
plot(features_all(idx_2d, 1*2-1), features_all(idx_2d, 1*2), 'b.', 'MarkerSize', 10);
for i = 1:size(idx_2d, 1)
    hold on;
    text(features_all(idx_2d(i), 1*2-1), features_all(idx_2d(i), 1*2), ...
        sprintf('\\leftarrow %d-%d', i, idx_2d(i)));
end

%%
w_id = X(7:56);
% w_id = [1; zeros(49, 1)];
w_exp = X(57:end);
% w_exp = [1; zeros(24, 1)];
tmpt_face = tmprod(core_tensor_truncated, {w_id, w_exp}, [2, 3], 'T');
tmpt_face = reshape(tmpt_face, 3, 11510);
figure; plot_mesh(tmpt_face, faces);
% plot3(tmpt_face(1, :), tmpt_face(2, :), tmpt_face(3, :), '.');
camlight('headlight');


tr = cal_tr(X);
mapped_face = tr * [tmpt_face; ones(1, 11510)];
mapped_face = [mapped_face(1, :) ./ mapped_face(3, :); mapped_face(2, :) ./ mapped_face(3, :)];
figure; imshow(pic_11); hold on; 
plot(mapped_face(1, corr_new_with_eyebrow(:, 2)), mapped_face(2, corr_new_with_eyebrow(:, 2)), '.');
% hold on; plot(features_0516(:, 17*2-1), features_0516(:, 17*2), 'r.');

%%
figure;
tmpt_face_idx = find(face_openmouth(:, 3) > 0);
tmpt_back_idx = find(face_openmouth(:, 3) <= 0);
plot3(face_openmouth(tmpt_face_idx, 1), face_openmouth(tmpt_face_idx, 2), face_openmouth(tmpt_face_idx, 3), 'r.');
hold on;
plot3(face_openmouth(tmpt_back_idx, 1), face_openmouth(tmpt_back_idx, 2), face_openmouth(tmpt_back_idx, 3), 'b.');


%%
for i = 4
    fprintf('displaying the result for %d pic...\n', i);
    display_optim_result(core_tensor_truncated, faces, ...
        img_all{i}, first_step_optim_result{i}.X, corr_all, features_0516(:, img_using(i)*2-1:img_using(i)*2), ...
        first_step_optim_result{i}.FVAL);
end

%%
tr = zeros(length(img_using), 6);
for i = 1:length(img_using)
    tr(i, :) = first_step_optim_result{i}.X(1:6);
end

%% using features come with db
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
features_tester_110 = zeros(74, 2);
fid = fopen('Tester_110/TrainingPose/pose_6.land');
l = fgetl(fid); num = str2num(l(1));
for i = 1:74
    l = fgetl(fid);
    features_tester_110(i, :) = sscanf(l, '%f %f');
end
fclose(fid);

close all
img_tester_110 = imread('Tester_110/TrainingPose/pose_6.png');
imshow(img_tester_110);
hold on; plot(640-tester_110(:, 1).*640, 480-tester_110(:, 2).*480, '.', 'MarkerSize', 10); axis on


%%
figure;
for i = 1:16
    subplot(4, 4, i);
    plot_mesh(reshape(a(:, i), 3, 11510), faces); camlight('headlight');
end