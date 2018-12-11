classdef xnat_assessor < handle    
    properties (Access = protected) % Possibly inherit for different assessors
        project
        subject
        session
        scan
        proc_type
    end
    
    methods (Access = private)        
        function label = get_label(obj)
            delimiter = '-x-';
                 
            if isempty(obj.get_scan())
                label = strjoin({obj.get_project(),obj.get_subject(),obj.get_session(),obj.get_proc_type()},delimiter);  
            else
                label = strjoin({obj.get_project(),obj.get_subject(),obj.get_session(),obj.get_scan(),obj.get_proc_type()},delimiter);  
            end     
        end     
        
        function assessor_URI = get_URI(obj)    
            % Note that assessors are stored at the session level
                        
            assessor_URI = fullfile(xnat_utils.get_session_URI(obj.get_project(),obj.get_subject(),obj.get_session()),'assessors',obj.get_label());
        end
                        
        function resource_URI = get_resource_URI(obj, resource)   
            % Resources for download are in the "out" resources
            
            resource_URI = fullfile(obj.get_URI(),'out','resources',resource);
        end
    end
    
    methods (Access = public)
        function obj = xnat_assessor(varargin)
            obj.project = varargin{1};
            obj.subject = varargin{2};
            obj.session = varargin{3};
            if nargin == 4
                obj.proc_type = varargin{4};
            elseif nargin == 5
                obj.scan = varargin{4};
                obj.proc_type = varargin{5};
            else
                error('Only 4 or 5 arguments allowed for xnat_assessor class.');
            end
        end
       
        function project = get_project(obj)
            project = obj.project;
        end
        
        function subject = get_subject(obj)
            subject = obj.subject;
        end
        
        function session = get_session(obj)
            session = obj.session;
        end
        
        function scan = get_scan(obj)
            if isempty(obj.scan)
                error(['Attempted to use get_scan() from assessor, but assessor: ' obj.get_label() ' is not a "scan assessor"']);
            end
            
            scan = obj.scan;
        end        
       
        function proc_type = get_proc_type(obj)
            proc_type = obj.proc_type;
        end
         
        % Query methods --------------------------------------------------%
        
        function status = exists(obj, xnat)
            status = xnat.xnat.select(obj.get_URI()).exists();
        end
        
        % Download methods -----------------------------------------------%
                          
        function get_resource(obj, xnat, resource, local_path)           
            xnat.get_resource(obj.get_resource_URI(resource),local_path);
        end
        
        function get_resource_file(obj, xnat, resource, resource_path, local_path)      
            xnat.get_resource_file(obj.get_resource_URI(resource),resource_path,local_path);
        end
                
        function get_single_file_resource(obj, xnat, resource, local_path)   
            % If a resource only has a single file, you can use this method
            % to download it and rename it. The benefits of using this
            % method is you do not need to know the name of the file in the
            % resource.
            
            % Get temporary name to download resource into
            tmp_path = tempname();
            
            % Retrieve contents of resource into temporary folder      
            obj.get_resource(xnat,resource,tmp_path);
            
            % Check to make sure resource only has one file
            l = dir(tmp_path);            
            if length(l) ~= 3 % listing includes "." and ".."
                % TODO: add cleanup for files retrieved
                error(['Either more than one file, or no files were found in ' resource ' resource. Expected a single file.']);
            end
            
            % Move file to local_path
            [status,message,~] = movefile(fullfile(tmp_path,l(3).name),local_path);
            if ~status
                % TODO: add cleanup for failed move
                error(['Failed to move ' resource ' file from temp folder to local_path. Reason: ' message]);
            end
                
            % Remove temporary directory
            [status,message,~] = rmdir(tmp_path,'s');
            if ~status
                error(['Failed to remove temporary directory while getting ' resource ' from XNAT. Reason: ' message]);
            end
        end                
    end    
end