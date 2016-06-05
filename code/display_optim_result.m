function display_optim_result(core_tensor, faces, pic, X, corr, feature)
%DISPLAY_OPTIM_RESULT Reconstructs a face mesh based on given identity
%weights and expression weights, and displays it.
%   Input:
%       core_tensor : 34530*50*25
%
%       faces : 22800*3; faces of a face mesh
%
%       pic : 640*480; the image which the face mesh corresponds to
%
%       X : 1*81; x(7:56) represents the identity weights, x(57:81) 
%       represents the expression weights
%
%       corr : n*2; correspondence between n 2D features and n 3D mesh
%       vertices. corr(:, 1) represents the index of 2D features, corr(:, 2)
%       represents the index of 3D mesh vertices
%
%       feature : 79*2; labelled features for a 2D image

    % generate face mesh
    w_id = X(7:56);
    w_exp = X(57:end);
    tmpt_face = tmprod(core_tensor, {w_id, w_exp}, [2, 3]);
    tmpt_face = reshape(tmpt_face, 3, 11510);

    figure;
    subplot(1, 2, 1);
    plot_mesh(tmpt_face, faces); camlight('headlight');
    % plot3(tmpt_face(1, :), tmpt_face(2, :), tmpt_face(3, :), '.');
    hold on; plot3(tmpt_face(1, corr(:, 2)), tmpt_face(2, corr(:, 2)), tmpt_face(3, corr(:, 2)), '.');
    
    % calculate mapped face
    tr = cal_tr(X);
    mapped_face = tr * [tmpt_face; ones(1, 11510)];
    mapped_face = [mapped_face(1, :) ./ mapped_face(3, :); mapped_face(2, :) ./ mapped_face(3, :)];
    
    subplot(1, 2, 2);
    imshow(pic);
    hold on; plot(mapped_face(1, corr(:, 2)), mapped_face(2, corr(:, 2)), 'b.');
    hold on; plot(feature(:, 1), feature(:, 2), 'r.');
    
    % calculate the distance
    delta = mapped_face(:, corr(:, 2)) - feature(corr(:, 1), :)';
    distance = sum(diag(delta' * delta)); 
    
    title(distance); drawnow;
end