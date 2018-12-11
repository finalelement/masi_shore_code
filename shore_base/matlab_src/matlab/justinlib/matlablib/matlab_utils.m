classdef matlab_utils < handle
% Library for commonly used matlab utilities

    methods (Static, Access = public)    
        
        % Structure related library --------------------------------------%        
        
        function s = add_unique_field(s, field, val)
            % Adds field to struct s and sets its value to val. Returns an 
            % error if field already exists.

            if ~isfield(s,field)
                s.(field) = val;
            else
                error(['Attempted to add field: ' field ' to struct, but it was not unique.']);
            end        
        end
        
        function s_merged = merge_structs(s1, s2)
            % Merges two input structures. Returns an error if one of the 
            % fields is not unique.

            % Initialize merged structure to s1
            s_merged = s1;

            fields_s2 = fieldnames(s2);
            for i = 1:length(fields_s2)
                s_merged = matlab_utils.add_unique_field(s_merged,fields_s2{i},s2.(fields_s2{i}));
            end
        end
        
        function s = replace_field(s, f1, f2)          
            % Replaces field f1 in s with f2
            val = s.(f1);
            s = rmfield(s,f1);
            s.(f2) = val;
        end
        
        function s = add_prefix(s, prefix)
            % Adds a prefix to input structure fields
            
            fields = fieldnames(s);
            for i = 1:length(fields)
                % Replace field with prefix'd field
                s = matlab_utils.replace_field(s,fields{i},[prefix fields{i}]);
            end            
        end        
    end    
end