function s = simplify_mesh(S, vid, tid)

s.surface.X = S.surface.X(vid); 
s.surface.Y = S.surface.Y(vid); 
s.surface.Z = S.surface.Z(vid); 
s.surface.TRIV = tid; 
s.surface.VERT = [s.surface.X, s.surface.Y, s.surface.Z]; 
s.surface.nv = length(s.surface.X); 