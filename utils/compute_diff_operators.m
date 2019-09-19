function [D_area, D_conf, D_ext] = compute_diff_operators(S_base, S_target, map, ndim_S, ndim_T)


if nargin < 4
    ndim_S = size(S_base.evecs, 2); 
    ndim_T = size(S_target.evecs, 2); 
end

if nargin < 3
    map = double(1:S_base.nv)'; 
end

C = S_target.evecs(:, 1:ndim_T)\S_base.evecs(map, 1:ndim_S); 

Phi = S_base.evecs(:, 1:ndim_S); 
X_base = S_base.surface.VERT; 
X_target = S_target.surface.VERT; 

D_base = squareform(pdist(X_base)).^2; 
D_target = squareform(pdist(X_target)).^2;  

D_base = S_base.A*D_base*S_base.A; 
D_target = S_target.A*D_target*S_target.A; 

L_base = diag(sum(D_base)) - D_base; 
L_target = diag(sum(D_target)) - D_target; 

inv_Lambda_base = [0; S_base.evals(2:ndim_S).^(-1)]; 


D_area = C'*C; 
D_conf = diag(inv_Lambda_base)*(C'*diag(S_target.evals(1:ndim_T))*C); 
D_ext = pinv(Phi'*L_base*Phi)*(C'*S_target.evecs(:, 1:ndim_T)'*L_target*S_target.evecs(:, 1:ndim_T)*C);

end