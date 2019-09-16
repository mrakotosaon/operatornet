%% Set path
addpath(genpath('./'));  


%% Load shapes and compute eigen stuff

% read the base shape (for the trained model).
S_base = compute_laplacian_basis(read_off_shape('./Data/Meshes/base.off'), 60); 

% for the compatibility with our trained model, we load the pre-computed eigenbasis for the base shape. 
fixed_basis = load('./Data/AuxData/BaseBasisFunc.mat'); 
S_base.evecs = fixed_basis.EVECS; 

% load two shapes for interpolation demo
S1 = read_obj('./Data/Meshes/50026_jiggle_on_toes3.obj'); 
S2 = read_obj('./Data/Meshes/50026_jiggle_on_toes31.obj'); 

% our shapes are obtained by a uniform simplification applied on the DFAUST/SMPL shapes (which share the same triangulation).
simp = load('./Data/AuxData/MeshSimplificationInfo.mat'); 
vid = simp.vid; % load the simplification -- vertex ids
tid = simp.tid; % load the simplification -- triangle ids

% simplify the meshes: before -> 6890 vts + 13776 faces; after -> 1000 vts
% + 1996 faces;
S1 = simplify_mesh(S1, vid, tid); 
S2 = simplify_mesh(S2, vid, tid); 

% compute the eigenbasis of the simpified meshes
S1 = compute_laplacian_basis(S1, 120); 
S2 = compute_laplacian_basis(S2, 120); 

%% Compute Difference Operators
% initialize the diff operators
D_area = cell(2, 1); 
D_conf = D_area; 
D_ext = D_area; 
% compute the diff operators for the two shapes to be interpolated
[D_area{1}, D_conf{1}, D_ext{1}] = compute_diff_operators(S_base, S1); 
[D_area{2}, D_conf{2}, D_ext{2}] = compute_diff_operators(S_base, S2); 
% save the output operator matrices
diff_maps = cell2mat(D_area); 
save('./Data/ShapeOperators/AreaSD.mat', 'diff_maps', '-v4'); 

diff_maps = cell2mat(D_conf); 
save('./Data/ShapeOperators/ConfSD.mat', 'diff_maps', '-v4'); 

diff_maps = cell2mat(D_ext); 
save('./Data/ShapeOperators/ExtSD.mat', 'diff_maps', '-v4'); 