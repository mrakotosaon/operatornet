%% Set path
addpath(genpath('./'));  

mkdir('./results'); 
mkdir('./results/groundtruth'); 
%% Load shapes and compute eigen stuff

% read the base shape (for the trained model).
S_base = compute_laplacian_basis(read_off_shape('./data/meshes/base.off'), 60); 

% for the compatibility with our trained model, we load the pre-computed eigenbasis for the base shape. 
fixed_basis = load('./data/auxdata/BaseBasisFunc.mat'); 
S_base.evecs = fixed_basis.EVECS; 

% load test shapes (5 in total: 3 for shape analogy and 2 for shape interpolation
shapes = dir('./data/meshes/*.obj');
% our shapes are obtained by a uniform simplification applied on the DFAUST/SMPL shapes (which share the same triangulation).
simp = load('./data/auxdata/MeshSimplificationInfo.mat'); 
vid = simp.vid; % load the simplification -- vertex ids
tid = simp.tid; % load the simplification -- triangle ids

S = cell(length(shapes), 1); 
for i = 1:5
    % read original obj files
    S{i} = read_obj(['./data/meshes/' shapes(i).name]); 
    % simplify the meshes: before -> 6890 vts + 13776 faces; after -> 1000 vts
    S{i} = simplify_mesh(S{i}, vid, tid); 
    S{i} = compute_laplacian_basis(S{i}, 120); 
    % save GT as ply files.
    plywrite(['./results/groundtruth/' shapes(i).name(1:end-4) '.ply'], S{i}.surface.TRIV, S{i}.surface.VERT);     
end



%% Compute Difference Operators
% initialize the diff operators
D_area = cell(length(shapes), 1); 
D_conf = D_area; 
D_ext = D_area; 
% compute the diff operators for the two shapes to be interpolated
for i = 1:length(shapes)
    [D_area{i}, D_conf{i}, D_ext{i}] = compute_diff_operators(S_base, S{i}); 
end

mkdir('./data/shapeoperators'); 


% save the output operator matrices
diff_maps = cell2mat(D_area); 
save('./data/shapeoperators/AreaSD.mat', 'diff_maps', '-v4'); 

diff_maps = cell2mat(D_conf); 
save('./data/shapeoperators/ConfSD.mat', 'diff_maps', '-v4'); 

diff_maps = cell2mat(D_ext); 
save('./data/shapeoperators/ExtSD.mat', 'diff_maps', '-v4'); 