function EXP = cal_blendshapes(core_tensor, faces, U_exp, w_id, idx)
%CAL_BLENDSHAPES Calculates the blendshapes based on given identity weights
%   Input:
%       core_tensor : 34530*50*25
%
%       faces : 22800*3; faces of a face mesh
%
%       U_exp : 47*25; transform matrix for expression mode
%
%       w_id : 1*50; identity weights
%
%       idx : 1*n, 1 <= n <= 47; indexes of desired blendshapes,
%       min(idx) >= 1, max(idx) <= 47

%   Output:
%       EXP : 34530*n; each column represents a face mesh vectorized to a
%       column vector

    EXP = zeros(34530, length(idx));
    for i = 1:length(idx)
        EXP(:, i) = tmprod(core_tensor, {w_id, -U_exp(idx(i), :)}, [2, 3]);
        figure; plot_mesh(reshape(EXP(:, i), 3, 11510), faces); camlight('headlight');
    end
end