function E = cal_energy_first_step(x, core_tensor, feature, corr)
%CAL_ENERGY_FIRST_STEP Calculates the energy in the first step optimization.
%   Input:
%       x : 1*81; x(7:56) represents the identity weights, x(57:81) 
%       represents the expression weights
%
%       core_tensor : n*50*25
%
%       feature : 79*2; labelled features for a 2D image
%
%       corr : n*2; correspondence between n 2D features and n 3D mesh
%       vertices. corr(:, 1) represents the index of 2D features, corr(:, 2)
%       represents the index of 3D mesh vertices
%
%   Output:
%       E : scalar
%
%   Description:
%       This script calculates the energy of formula (3)      

    % calculate transformation and projection
    tr = cal_tr(x);

    % generate face mesh
    face_reconstructed = tmprod(core_tensor, {x(7:56), x(57:end)}, [2, 3]);
    face_reconstructed = reshape(face_reconstructed, 3, size(corr, 1));

    % calcualte the mapped face
    transformed_face = tr * [face_reconstructed; ones(1, size(corr, 1))];
    mapped_face = [transformed_face(1, :) ./ transformed_face(3, :); transformed_face(2, :) ./ transformed_face(3, :)];

    % calculate engery
    delta = mapped_face - feature(corr(:, 1), :)';
    E = sum(diag(delta' * delta));
end