classdef directory < handle
% directory class to deal with directories

    properties (Access = private)
        root_dir_path
        parent_dir
        name
    end
    
    methods (Static, Access = private)
        function varargout = get_dir_inputs(varargin)               
            % Last input argument is a string describing the operation.
            % Rest of the input arguments are specified below.
            %
            % First output argument is either a directory object to the
            % folder containing this directory, or a path to the folder
            % containing this directory. The second output argument is the
            % name of the directory. The third output argument is the full
            % path to the directory.
                        
            if nargin == 2
                % First argument is the absolute path to this directory
                [varargout{1},dir_name_1,dir_name_2] = fileparts(varargin{1}); 
                varargout{2} = [dir_name_1 dir_name_2]; % dir_name_2 can happen if directory name has a period in it
                varargout{3} = varargin{1};
            elseif nargin == 3
                % First argument is either the absolute path to the folder 
                % containing this directory, or a directory object to the
                % folder containing this directory. The second argument is 
                % the name of this directory.
                varargout(1:2) = varargin(1:2);
                if isa(varargin{1},'directory')
                    varargout{3} = fullfile(varargin{1}.get_path(),varargin{2});
                else
                    varargout{3} = fullfile(varargin{1},varargin{2});
                end
            else
                error(['Only 1 or 2 arguments may be specified for ' ...
                    varargin{end} '. If one argument is specified, ' ...
                    'then it is the full path to this directory; ' ...
                    'this directory is considered a "root" directory. ' ...
                    'If two arguments are specified, then the first ' ...
                    'argument must be a "parent" directory object to ' ...
                    'the folder containing this directory, or a path ' ...
                    'to the folder containing this directory. If a ' ...
                    'path is specified, then this directory is ' ...
                    'considered a "root" directory. The second argument ' ...
                    'must be the directory name.']);
            end  
        end
    end
          
    methods (Access = private)        
        function set_props(obj, d, n)            
            % d is either a directory object or an absolute path to the
            % folder containing this directory. n is the name of the
            % directory.
            
            if isa(d,'directory')
                obj.root_dir_path = [];
                obj.parent_dir = d;
            else
                obj.root_dir_path = d;
                obj.parent_dir = [];
            end          
            obj.name = n;
        end
    end
    
    methods (Access = public)
        function obj = directory(varargin) 
            [d,n,~] = directory.get_dir_inputs(varargin{:}, 'directory class'); 
            
            obj.set_props(d,n);
        end
        
        function parent_dir = get_parent_dir(obj)
            parent_dir = obj.parent_dir; % Returns empty for "root" directory
        end
        
        function status = is_root_dir(obj)
            status = isempty(obj.get_parent_dir());
        end
        
        function dir_path = get_dir_path(obj)
            if obj.is_root_dir()
                dir_path = obj.root_dir_path;
            else
                dir_path = obj.get_parent_dir().get_path();
            end
        end
        
        function name = get_name(obj)
            name = obj.name;
        end
        
        function path = get_path(obj)
            path = fullfile(obj.get_dir_path(),obj.get_name());
        end
        
        function status = exist(obj)
            % Returns true if directory exists
            
            status = exist(obj.get_path(),'dir') ~= 0;
        end
        
        % ZIP methods ----------------------------------------------------%
                
        function zipped_folder = zip(obj, zipped_name)
            % zips a folder and returns a file object in the parent
            % directory.
            
            % Check to make sure directory exists            
            if ~obj.exist()
                error(['Attempted to zip folder: ' obj.get_path() ', but this folder does not exist.']);
            end
                        
            % Create name if it wasn't supplied - default to directory name          
            if ~exist('zipped_name','var')
                zipped_name = [obj.get_name() '.zip'];
            end
                        
            % Create a new file object
            if obj.is_root_dir()
                zipped_folder = file(obj.get_dir_path(),zipped_name);   
            else
                zipped_folder = file(obj.get_parent_dir(),zipped_name);   
            end
            
            % zip folder
            zip(zipped_folder.get_path(),obj.get_path());
        end
        
        % TAR methods ----------------------------------------------------%
        
        function tarred_folder = tar(obj, tarred_name)
            % tars a folder and returns a file object in the parent
            % directory.
            
            % Check to make sure directory exists            
            if ~obj.exist()
                error(['Attempted to tar folder: ' obj.get_path() ', but this folder does not exist.']);
            end
               
            % Create name if it wasn't supplied - default to directory name           
            if ~exist('tarred_name','var')
                tarred_name = [obj.get_name() '.tar.gz']; % As of R2015a '.tgz' extension does not work correctly. 
            end
                        
            % Create a new file object
            if obj.is_root_dir()
                tarred_folder = file(obj.get_dir_path(),tarred_name);
            else
                tarred_folder = file(obj.get_parent_dir(),tarred_name);   
            end              
            
            % tar folder
            tar(tarred_folder.get_path(),obj.get_path());
        end        
        
        % System commands ------------------------------------------------%
        
        function mkdir(obj)  
            % Makes directory. If directory already exists, then this does
            % nothing.
            
            [status,message,~] = mkdir(obj.get_path());
            if ~status
                error(['Attempted to create ' obj.get_path() ' directory, but failed. Reason: ' message]);
            end 
        end
        
        function contents = dir(obj,name)
            % Performs dir operation and returns files and directories.
            % Makes the parent directory this obj by default. Also removes
            % directories with name '.' and '..'.
            contents = struct('files',file.empty(),'directories',directory.empty());
            
            dir_path = obj.get_path();
            if exist('name','var')
                % Merge them
                dir_path = fullfile(dir_path,name);
            end
            
            % Get listing and remove directories with '.' and '..'
            l = dir(dir_path);
            l([l.isdir] & strcmp('.',{l.name})) = [];
            l([l.isdir] & strcmp('..',{l.name})) = [];
            
            for i = 1:length(l)
                if l(i).isdir
                    contents.directories(end+1) = directory(obj,l(i).name);
                else
                    contents.files(end+1) = file(obj,l(i).name);                    
                end                   
            end
        end
        
        function mv(obj, varargin)
            % Moves existing directory to location specified by input(s).
                        
            [d,n,new_path] = directory.get_dir_inputs(varargin{:}, 'mv() method'); 
            
            % Check to make sure directory actually exists first
            if ~obj.exist()
                error(['Attempted to move directory: ' obj.get_path() ' to path: ' new_path ', but this directory does not exist.']);
            end 
                       
            % Check if this directory and destination has the same path
            if strcmp(obj.get_path(),new_path)
                return % Do nothing
            end
                        
            % Now, move directory
            [status,message,~] = movefile(obj.get_path(),new_path);
            if ~status
                error(['Attempted to move directory: ' obj.get_path() ' to path: ' new_path ', but failed. Reason: ' message]);
            end
                                              
            % Set new properties           
            obj.set_props(d,n);
        end
                    
        function copied_dir = cp(obj, varargin)
            % Copies existing directory to location specified by input(s).
            % Returns a new directory object to the copied directory.
                             
            [d,n,new_path] = directory.get_dir_inputs(varargin{:}, 'cp() method'); 
                                    
            % Check to make sure directory actually exists first
            if ~obj.exist()
                error(['Attempted to copy directory: ' obj.get_path() ' to path: ' new_path ', but this directory does not exist.']);
            end
            
            % Create new directory object
            copied_dir = directory(d,n);
            
            % Check if this directory and destination has the same path
            if strcmp(obj.get_path(),new_path)
                return % Do nothing
            end
            
            % Now, copy directory
            [status,message,~] = copyfile(obj.get_path(),new_path);
            if ~status
                error(['Attempted to copy directory: ' obj.get_path() ' to path: ' new_path ', but failed. Reason: ' message]);
            end
        end
        
        function rm(obj)           
            % Removes directory and its contents, so be careful.
                        
            [status,message,~] = rmdir(obj.get_path(),'s');    
            if ~status
                error(['Attempted to remove ' obj.get_path() ' directory, but failed. Reason: ' message]);
            end   
        end        
    end
end