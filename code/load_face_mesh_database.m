function load_face_mesh_database(path_dir)
%LOAD_FACE_MESH_DATABASE loads the raw .obj files of face meshs, and save
%them to .mat files.
%   Input:
%       path_dir : string; root path of the database

    % load expressions for 20 person, each with 5 expressions
    num_vertice = 11510;
    num_person = 100;
    num_expression = 47;
    
    data = zeros(num_vertice*3, num_person, num_expression);

    for i = 51:50+num_person
        fprintf('loading facial models for %d-th person...\n', i);
        for j = 1:num_expression
            file_name = sprintf('%s/Tester_%d/Blendshape/shape_%d.obj', path_dir, i, j-1);
            data(:, i-50, j) = reshape(load_obj(file_name)', num_vertice*3, 1, 1);
        end
    end
    
    save('database_51_150.mat', 'data');

    function face_obj = load_obj(file_name)
        fid = fopen(file_name);
        frewind(fid);
        face_obj = zeros(11510, 3);
        for ii = 1:11510
            l = fgetl(fid);
            face_obj(ii, :) = sscanf(l(3:end), '%f %f %f');
        end
        fclose(fid);
    end
end