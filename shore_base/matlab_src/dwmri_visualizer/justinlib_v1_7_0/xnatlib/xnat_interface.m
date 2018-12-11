classdef xnat_interface < handle
    properties (Access = public) % Making public allows users flexibility to use any URI
        xnat 
    end
    
    methods (Access = public)
        function obj = xnat_interface(XNAT_HOST, XNAT_USER, XNAT_PASS)
            % Uses environment variables to form connection if inputs are 
            % not supplied.

            if ~exist('XNAT_HOST','var')
                XNAT_HOST = system_utils.getenv_with_errorcheck('XNAT_HOST');
            end

            if ~exist('XNAT_USER','var')
                XNAT_USER = system_utils.getenv_with_errorcheck('XNAT_USER');
            end

            if ~exist('XNAT_PASS','var')
                XNAT_PASS = system_utils.getenv_with_errorcheck('XNAT_PASS');
            end
            
            % Get connection   
            disp('---');
            disp('Getting XNAT connection...');
            obj.xnat = py.pyxnat.Interface(XNAT_HOST,XNAT_USER,XNAT_PASS);
        end       
        
        % Download methods -----------------------------------------------%
                        
        function get_resource(obj, resource_URI, local_path)
            % Attempts to download entire resource located at resource_URI 
            % to folder specified in local_path. Resources on xnat are sent
            % as a zip. So you must create the folder, get the zip, and 
            % then unzip the file in the folder.
            
            if ~obj.xnat.select(resource_URI).exists()
                error(['Resource: ' resource_URI ' does not exist on XNAT.']);
            end
            
            % Get path of zip on local_path
            [~,resource_name] = fileparts(resource_URI);
            zip_path = fullfile(local_path,[resource_name '.zip']);
            
            % Throw a warning if directory already exists and an error if
            % a zip with the same name happens to exist in it. 
            % TODO: Possibly use tempname to store zip instead.
            if exist(local_path,'dir')
                disp(['Warning: folder: ' local_path ' already exists. Files from resource will be merged with files in this directory.']);
                if exist(zip_path,'file')         
                    error(['zip file: ' zip_path ' already exists in folder: ' local_path '. Move or delete this zip if you want to download this resource.']);
                end
            else
                [status,message,~] = mkdir(local_path);
                if ~status
                    error(['Failed to create directory while downloading resource: ' resource_URI ' from XNAT. Reason: ' message]);
                end
            end         

            % Download resource
            obj.xnat.select(resource_URI).get(local_path);

            % Check to see if zip file was actually retrieved 
            if ~exist(zip_path,'file')         
                error(['zip file: ' zip_path ' cannot be found while attempting to download resource from XNAT. Are you sure the resource URI: ' resource_URI ' was correct?']);
            end

            % Unzip file
            unzip(zip_path,local_path);

            % Remove zip file
            delete(zip_path);
        end 
        
        function get_resource_file(obj, resource_URI, resource_path, local_path)
            % Attempts to download file located at resource_path within the
            % resource_URI and places it at local_path.
            
            if ~obj.xnat.select(resource_URI).file(resource_path).exists()
                error(['File: ' fullfile(resource_URI,'files',resource_path) ' does not exist on XNAT.']);
            end
            
            obj.xnat.select(resource_URI).file(resource_path).get(local_path);
        end 
        
        function delete(obj)
            if (~isempty(obj.xnat)) % Only close connection if xnat has been set
                obj.xnat.disconnect();
                disp('---');
                disp('XNAT connection closed');
            end
        end
    end    
end