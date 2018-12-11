classdef xnat_scan < handle    
    properties (Access = protected) % Possibly inherit for different scans
        xnat
        
        project
        subject
        session
        scan
    end
    
    methods (Access = private)        
        function scan_path = get_scan_path(obj)   
            scan_path = fullfile(xnat_utils.get_session_path(obj.get_project(),obj.get_subject(),obj.get_session()),'scans',obj.get_scan());
        end
        
        function xnat_folder_path = get_xnat_folder_path(obj, folder)   
            % Returns path to select scan folder on XNAT
            
            xnat_folder_path = fullfile(obj.get_scan_path(),'resources',folder);
        end 
    end
        
    methods (Access = public)
        function obj = xnat_scan(xnat, project, subject, session, scan)
            obj.xnat = xnat;
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
        
        function status = exists(obj)
            status = obj.xnat.exists(obj.get_scan_path());
        end
        
        % Download methods -----------------------------------------------%
        
        function get_file(obj, xnat_folder, xfp, xgp)                     
            obj.xnat.get_file(obj.get_xnat_folder_path(xnat_folder),xfp,xgp);
        end
        
        function get_folder(obj, xnat_folder, xgp)                      
            obj.xnat.get_folder(obj.get_xnat_folder_path(xnat_folder),xgp);
        end        
                      
        function get_single_file_from_folder(obj, xnat_folder, xgp)   
            % Get temporary name to download folder into
            tmp_path = tempname();
            
            % Retrieve contents of folder into temporary folder
            obj.get_folder(xnat_folder,tmp_path);
            
            % Check to make sure folder only has one file
            l = dir(tmp_path);            
            if length(l) ~= 3 % listing includes "." and ".."
                % TODO: add cleanup for files retrieved
                error(['Either more than one file, or no files were found in ' xnat_folder ' folder. Expected a single file.']);
            end
            
            % Move file to xgp
            [status,message,~] = movefile(fullfile(tmp_path,l(3).name),xgp);
            if ~status
                % TODO: add cleanup for failed move
                error(['Failed to move ' xnat_folder ' file from temp folder to input path. Reason: ' message]);
            end
                
            % Remove temporary directory
            [status,message,~] = rmdir(tmp_path,'s');
            if ~status
                error(['Failed to remove temporary directory while getting ' xnat_folder ' from XNAT. Reason: ' message]);
            end
        end
        
        % Upload methods -------------------------------------------------%
        function upload_file(obj, xpfp, xnat_folder)                      
            obj.xnat.upload_file(obj.get_xnat_folder_path(xnat_folder),xpfp);
        end           
        
        function upload_folder(obj, xpfp, xnat_folder)                      
            obj.xnat.upload_folder(obj.get_xnat_folder_path(xnat_folder),xpfp);
        end           
    end    
end