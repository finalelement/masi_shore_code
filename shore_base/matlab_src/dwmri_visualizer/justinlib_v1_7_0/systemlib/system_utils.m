classdef system_utils < handle
% Library for commonly used system utilities

    methods (Static, Access = public)                
        function env_val = getenv_with_errorcheck(env_name)
            % Returns environmental variable specified by env_name. This 
            % assumes that if the environmental variable is empty, that it 
            % has not been defined on the system and will return an error.

            env_val = getenv(env_name);
            if isempty(env_val) 
                error([env_name ' environmental variable is not set.']);
            end
        end 
        
        function cmdout = system_with_errorcheck(cmd, err_msg)
            disp(['[' char(datetime) '] ' cmd]);

            [status,cmdout] = system(cmd,'-echo');
            if status
                error(err_msg);
            end
        end
    end    
end