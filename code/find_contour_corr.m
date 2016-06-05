function [corr, dist] = find_contour_corr(im, feature, core_tensor, X)
%FIND_CONTOUR_CORR finds the correspondence between 2D face contour
%features and the vertices of 3D face mesh, based on the correspondece of
%inner face features.
%   Input:
%       im : 640*480
%
%       feature : 79*2; labelled features on im
%
%       core_tensor : 34530*50*25
%
%       X : 1*81; x(7:56) represents the identity weights, x(57:81) 
%       represents the expression weights
%
%   Output:
%       corr : 19*2; first column is the indexes of 2D features, second
%       column is the indexes of 3D vertices
%
%       dist : 19*1; the nearest distance of 2D face contour features to
%       mapped 3D vertices

    % calculate transformation and perspective projection
    tr = cal_tr(X);
    
    % calcualte the mesh
    w_id = X(7:56); w_exp = X(57:81);

    face_mesh = tmprod(core_tensor, {w_id, w_exp}, [2, 3]);
    face_mesh = reshape(face_mesh, 3, 11510)';
    
    % find face region
    face_region_idx = find(face_mesh(:, 3) > 0);
    
    face_region = face_mesh(face_region_idx, :);
    
    % transformation and projection
    transformed_face_region = tr * [face_region'; ones(1, length(face_region_idx))];
    mapped_face_region = [transformed_face_region(1, :)./transformed_face_region(3, :); ...
                          transformed_face_region(2, :)./transformed_face_region(3, :)];
    
    % find correspondence, i.e. 1. find corr in face region; 2. find corr
    % in original face mesh
    [corr_pre, dist] = knnsearch(mapped_face_region', feature(1:19, :), 'k', 1);
    [~, corr] = ismember(face_region(corr_pre, :), face_mesh, 'rows');

    corr = [[1:19]', corr];
    
%     figure; imshow(im);
%     hold on; plot(mapped_face_region(1, :), mapped_face_region(2, :), '.');
end

