function tr = cal_tr(x)
%CAL_TR Calculates the product of transformation matrix and perspective
%projection matrix.
%   Input:
%       x : 1*6 row vector or 6*1 column vector. First three elements are
%       rotation angles; last three elements are translation distance
%       
%   Output:
%       tr : 3*4 matrix
%
%   Description:
%       3D rotation matrix can be expressed as R = Rx*Ry*Rz, 
%       where Rx =  1       0         0
%                   0   cos(x(1))  -sin(x(1))
%                   0   sin(x(1))   cos(x(1))
%
%             Ry =  cos(x(2))     0     sin(x(2))
%                       0         1        0
%                   -sin(x(2))    0     cos(x(2))
%
%             Rz =  cos(x(3))  -sin(x(3))  0
%                   sin(x(3))  cos(x(3))   0
%                       0          0       1
%
%       Translation matrix is just:
%       [x(4); x(5); x(6)]
%
%       Perspective projection matrix looks like:
%       Q =  fx    0    u0
%            0     fy   v0
%            0     0    1

    % calculate rotation matrix
    Rx = [1 0 0; 0 cos(x(1)) -sin(x(1)); 0 sin(x(1)) cos(x(1))];
    Ry = [cos(x(2)) 0 sin(x(2)); 0 1 0; -sin(x(2)) 0 cos(x(2))];
    Rz = [cos(x(3)), -sin(x(3)) 0; sin(x(3)) cos(x(3)) 0; 0 0 1];
    
    % calculate translation matrix
    T = [x(4) x(5) x(6)]'; 
    
    % get transformation matrix
    R = Rz * Ry * Rx;
    M = [R T];
    
    % Using 580 as fx and fy in our project. I do not think this would
    % affect the optimization result. Since the optimization would adjust
    % to this automatically
    Q = [580 0 320; 0 580 240; 0 0 1];
    
    % return
    tr = Q * M;
end