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
                
        function chk_dirs_exist(dirs)
            % Assumes dirs is a structure
            cell_dirs = struct2cell(dirs);
            for i = 1:length(cell_dirs)
                if ~cell_dirs{i}.exist()
                    error(['Required directory: ' cell_dirs{i}.get_name() ' in: ' cell_dirs{i}.get_dir_path() ' cannot be found.']);
                end
            end
        end
        
        function chk_files_exist(files)
            % Assumes files is a structure
            cell_files = struct2cell(files);
            for i = 1:length(cell_files)
                if ~cell_files{i}.exist()
                    error(['Required file: ' cell_files{i}.get_name() ' in: ' cell_files{i}.get_dir_path() ' cannot be found.']);
                end
            end
        end
        
        function add_MATLAB_paths(MATLAB_paths)
            % Will add the paths in MATLAB_paths structure to matlab path.
            % Note that this will add to the FRONT of the matlab path, so 
            % functions specified in these paths will get called if there 
            % are conflicts.            
            cell_MATLAB_paths = struct2cell(MATLAB_paths);
            if isempty(cell_MATLAB_paths)
                return;
            end

            addpath(cell_MATLAB_paths{:});
        end

        function add_PATH_paths(PATH_paths)
            % Will add the paths in PATH_paths structure to the PATH 
            % environmental variable. Note that this will add to the FRONT 
            % of the PATH variable, so functions specified in these paths 
            % will get called if there are conflicts.            
            cell_PATH_paths = struct2cell(PATH_paths);
            if isempty(cell_PATH_paths)
                return;
            end

            % Get paths and split with colon
            paths = strsplit(getenv('PATH'),pathsep);

            % Loop over paths and add each one to front
            for i = 1:length(cell_PATH_paths)
                % Delete duplicates if there are any
                paths(strcmp(paths,cell_PATH_paths{i})) = [];

                % Append path to front
                paths = horzcat(cell_PATH_paths{i},paths);
            end

            % Join paths with colon 
            paths = strjoin(paths,pathsep);

            % Set path variable
            setenv('PATH',paths);
        end
        
        function system_with_errorcheck(cmd, err_msg)
            disp(['Running command: ' cmd]);

            status = system(cmd);
            if status
                error(err_msg);
            end
        end
    end    
end