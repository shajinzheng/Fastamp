function E = cal_energy_second_step(x, core_tensor, features, corr, W_EXP, TR)
%CAL_ENERGY_SECOND_STEP Calculates the energy in the second step optimization.
%   Input:
%       x : 1*50; the expression weights
%
%       core_tensor : n*50*25
%
%       features : 79*(2*m); labelled features for m 2D images
%
%       corr : n*2; correspondence between n 2D features and n 3D mesh
%       vertices. corr(:, 1) represents the index of 2D features, corr(:, 2)
%       represents the index of 3D mesh vertices
%
%       W_EXP : m * 25; each row is a set of expression weights of a mesh
%
%       TR : m * 12; each row is the transformation-projection matrix 
%
%   Output:
%       E : scalar
%
%   Description:
%       This script calculates the energy of formula (4)

    E = 0;
    M = size(W_EXP, 1);

    for i = 1:M
        % transformation and projection
        tr = reshape(TR(i, :), 3, 4);

        w_exp = W_EXP(i, :);
            
        % generate face mesh
        face_reconstructed = tmprod(core_tensor, {x, w_exp}, [2, 3]);
        face_reconstructed = reshape(face_reconstructed, 3, size(corr, 1));

        % calcualte the mapped face
        transformed_face = tr * [face_reconstructed; ones(1, size(corr, 1))];
        mapped_face = [transformed_face(1, :) ./ transformed_face(3, :); transformed_face(2, :) ./ transformed_face(3, :)];

        % calculate engery
        delta = mapped_face - features(corr(:, 1), i*2-1:i*2)';
        E = E + sum(diag(delta' * delta));
    end
end