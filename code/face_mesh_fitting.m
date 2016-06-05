function W = face_mesh_fitting(core_tensor, faces, corr, features, num_iter, pics)
%FACE_MESH_FITTING applies a two-step approach to find the best fit of the
%3D model for the specific user.
%   Input:
%       core_tensor : 34530*50*25
%
%       faces : 22800*3; faces of a face mesh 
%
%       corr : n*2; correspondence between n 2D features and n 3D mesh
%       vertices. corr(:, 1) represents the index of 2D features, corr(:, 2)
%       represents the index of 3D mesh vertices
%
%       features : 79*(2*m); labelled features for m 2D images
%
%       num_iter : scalar; the number of iterations in optim
%
%       pics : {1*m} cell
%
%   Output:
%       W : 1*1 struct; W.first_step is the optim results in first step;
%       W.second_step is the optim results in second step

    if size(features, 2)/2 ~= length(pics)
        error('The number of pics should be equal to the number of pictures we have labelled features on.');
    end

    num_pic = size(features, 2)/2;

    % calculate small core tensor
    fprintf('calculating small core tensor...\n');
    pick_idx = ones(size(corr, 1)*3, 1);
    for i = 1:size(corr, 1)
        pick_idx(i*3-2:i*3) = 3*corr(i, 2)-2:3*corr(i, 2);
    end
    small_core_tensor = core_tensor(pick_idx, :, :);
    
    % preallocate
    first_step = cell(1, num_iter+1);
    second_step = cell(1, num_iter+1);
    for i = 1:num_iter+1
        first_step{i}.X = zeros(num_pic, 81);
        first_step{i}.FVAL = zeros(num_pic, 1);
        
        second_step{i}.X = zeros(1, 50);
        second_step{i}.FVAL = 0;
    end
    
    % initialize
    first_step{1}.X(:, 7) = 1;
    first_step{1}.X(:, 57) = 1;
    second_step{1}.X = [1, zeros(1, 49)];
    
    % set optimization options
    options = optimoptions(@fmincon, 'TolX', 1e-6, 'Algorithm', 'sqp', ...
        'PlotFcn', {@optimplotx, @optimplotfval}, 'MaxFunEvals', 50000, 'OutputFcn', @outfun);    
    Aeq1 = [
            [zeros(1, 6), ones(1, 50), zeros(1, 25)]; ...
            [zeros(1, 6), zeros(1, 50), ones(1, 25)]
           ];
    Beq1 = [1; 1];
    LB1 = [-pi*ones(3, 1); -20*ones(3, 1); zeros(75, 1)];
    UB1 = [pi*ones(3, 1); 20*ones(3, 1); ones(75, 1)];
    Aeq2 = ones(1, 50);
    Beq2 = 1;
    LB2 = zeros(50, 1);
    UB2 = ones(50, 1);
    
    % optimize
    for i = 1:num_iter
        %%%%%%%%%%%%%%%
        % first step  %
        %%%%%%%%%%%%%%% 
        for j = 1:num_pic
            fprintf('In iter %d, first step optim for pic %d...\n', i, j);
            
            X0 = first_step{i}.X(j, :);
            X0(7:56) = second_step{i}.X;
            
            % use optim method provided by Matlab
            [first_step{i+1}.X(j, :), first_step{i+1}.FVAL(j), ~, ~] ...
                = fmincon(@(x)cal_energy_first_step(x, small_core_tensor, features(:, j*2-1:j*2), corr), ...
                X0, [], [], Aeq1, Beq1, LB1, UB1, [], options);
    
            % display result
            display_optim_result(core_tensor, faces, pics{j}, first_step{i+1}.X(j, :), ...
                corr, features(:, j*2-1:j*2));
            
            pause(1); close all;
        end
        
        %%%%%%%%%%%%%%%
        % second step %
        %%%%%%%%%%%%%%%        
        % use the first pic's identity weights to initialize
        X0 = first_step{i+1}.X(1, 7:56);

        % calculate expression weights
        W_EXP = first_step{i+1}.X(:, 57:end);
        
        % calcualte the product of transformation and projection matrixes
        TR = zeros(num_pic, 12);
        for k = 1:num_pic
            TR(k, :) = reshape(cal_tr(first_step{i+1}.X(k, 1:6)), 1, 12);
        end

        fprintf('In iter %d, second step optim...\n', i);
        
        % optimize
        [second_step{i+1}.X, second_step{i+1}.FVAL, ~, ~] = ...
            fmincon(@(x)cal_energy_second_step(x, small_core_tensor, features, corr, W_EXP, TR), ...
            X0, [], [], Aeq2, Beq2, LB2, UB2, [], options);        
        
        % display
        face_mesh = tmprod(core_tensor, {second_step{i+1}.X, [1 zeros(1, 24)]}, [2, 3]);
        figure; plot_mesh(reshape(face_mesh, 3, 11510), faces); camlight('headlight');
        
        pause(1); close all;
        
%         %%%%%%%%%%%%%%%%%%
%         % update contour %
%         %%%%%%%%%%%%%%%%%%
%         % calculate new contour feature correspondence
%         X = first_step{i+1}.X(1, :);
%         X(7:56) = second_step{i+1}.X;
%         [corr_contour, ~] = find_contour_corr(pics{1}, features(:, 1:2), core_tensor, X);
% 
%         % plot
%         figure;
%         mapped_face_1 = helpFunc(X, corr_contour);
%         mapped_face_2 = helpFunc(X, corr(end-18:end, :));
%         imshow(pics{1});
%         hold on; plot(mapped_face_1(1, :), mapped_face_1(2, :), 'r.');
%         hold on; plot(mapped_face_2(1, :), mapped_face_2(2, :), 'b.');
%         pause(1); close all;
%         
%         % update
%         corr(end-18:end, :) = corr_contour;
    end
   
    W = struct('first_step', first_step, 'second_step', second_step);
    
    
    function rr = helpFunc(x_local, corr_local)
        tmpt_face = tmprod(core_tensor, {x_local(7:56), x_local(57:end)}, [2, 3]);
        tmpt_face = reshape(tmpt_face, 3, 11510);
        
        tr = cal_tr(x_local);
        mapped_face_local = tr * [tmpt_face; ones(1, 11510)];
        mapped_face_local = [mapped_face_local(1, :) ./ mapped_face_local(3, :); mapped_face_local(2, :) ./ mapped_face_local(3, :)];
        
        rr = mapped_face_local(:, corr_local(:, 2));
    end
    
    function stop = outfun(x, optimValues, state)
        stop = false;
        if optimValues.fval < 2000 
            stop = true; 
        end
    end 
end