function [corr, dist] = find_eyebrow_corr(im, features, core_tensor, X)
    % calculate transformation and perspective projection
    tr = cal_tr(X);
    
    % calcualte the mesh
    w_id = X(7:56); w_exp = X(57:81);

    face_mesh = tmprod(core_tensor, {w_id, w_exp}, [2, 3], 'T');
    face_mesh = reshape(face_mesh, 3, 11510)';
    
    % find face region
    face_region_idx = find(face_mesh(:, 3) > 0);
    
    face_region = face_mesh(face_region_idx, :);
    
    % transformation and projection
    transformed_face_region = tr * [face_region'; ones(1, length(face_region_idx))];
    mapped_face_region = [transformed_face_region(1, :)./transformed_face_region(3, :); ...
                          transformed_face_region(2, :)./transformed_face_region(3, :)];
    
    % find correspondence
    % 1. find corr in face region; 
    % 2. find corr in original face mesh
    [corr_pre, dist] = knnsearch(mapped_face_region', features([28:35, 72:79], :), 'k', 1);
    [~, corr] = ismember(face_region(corr_pre, :), face_mesh, 'rows');
    corr = [[28:35, 72:79]', corr];
    
    figure; imshow(im);
    hold on; plot(mapped_face_region(1, :), mapped_face_region(2, :), '.');
end

