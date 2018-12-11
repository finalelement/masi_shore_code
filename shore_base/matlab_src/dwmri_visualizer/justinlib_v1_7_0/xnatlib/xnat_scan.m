classdef xnat_scan < handle    
    properties (Access = protected) % Possibly inherit for different scans
        project
        subject
        session
        scan
    end
    
    methods (Access = private)        
        function scan_URI = get_URI(obj)   
            scan_URI = fullfile(xnat_utils.get_session_URI(obj.get_project(),obj.get_subject(),obj.get_session()),'scans',obj.get_scan());
        end
        
        function resource_URI = get_resource_URI(obj, resource)   
            resource_URI = fullfile(obj.get_URI(),'resources',resource);
        end 
    end
        
    methods (Access = public)
        function obj = xnat_scan(project, subject, session, scan)
            obj.project = project;
            obj.subject = subject;
            obj.session = session;
            obj.scan = scan;
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
            scan = obj.scan;
        end
        
        % Query methods --------------------------------------------------%
        
        function status = exists(obj,xnat)
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