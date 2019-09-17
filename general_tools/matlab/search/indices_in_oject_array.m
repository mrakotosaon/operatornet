function [ index ] = indices_in_oject_array(object_array, property_name, boolean_f)
% Finds the indices of the objects that satisfy a boolean function that is applied to
% a particular property of each object. 
% This function is similar to findobj('-function') but returns the indices
% instead of pointers to the objects.

    n = length(object_array);
    bit_vector = false(n, 1);
    for i =1:length(object_array)
        bit_vector(i) = boolean_f(object_array(i).(property_name));
    end
    index = find(bit_vector);
end