classdef xnat_interface < handle
    properties (Access = private)
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
        
        % Query methods --------------------------------------------------%
        
        function status = exists(obj, xsp)
            status = obj.xnat.select(xsp).exists();
        end
        
        % Download methods -----------------------------------------------%
        
        function get_file(obj, xsp, xfp, xgp)
            % Attempts to download file from xnat and places it at the path
            % specified by xgp.
            %
            % xsp = xnat "select" path
            % xfp = xnat "file" path
            % xgp = xnat "get" path
            
            if ~obj.xnat.select(xsp).file(xfp).exists()
                error(['File: ' fullfile(xsp,'files',xfp) ' does not exist on XNAT.']);
            end
            
            obj.xnat.select(xsp).file(xfp).get(xgp);
        end 
                
        function get_folder(obj, xsp, xgp)
            % Attempts to download folder from xnat and places it in the
            % path specified by xgp. This is a little more complicated 
            % because folders on xnat are sent as a zip. So you must create 
            % the folder, get the zip, and then unzip the file in the 
            % folder.
            %
            % xsp = xnat "select" path
            % xgp = xnat "get" path
            
            if ~obj.xnat.select(xsp).exists()
                error(['Folder: ' xsp ' does not exist on XNAT.']);
            end
            
            % Create directory if it doesnt exist
            if ~exist('xgp','dir')
                [status,message,~] = mkdir(xgp);
                if ~status
                    error(['Failed to create directory while downloading folder: ' xsp ' from XNAT. Reason: ' message]);
                end
            end         

            % Download folder (will be a zip)
            % TODO: add error checking in case zip already exists in the
            % folder
            obj.xnat.select(xsp).get(xgp);

            % Check to see if zip file was actually retrieved 
            [~,folder_name] = fileparts(xsp);
            zip_path = fullfile(xgp,[folder_name '.zip']);
            if ~exist(zip_path,'file')
                % TODO: add clean-up to remove folder if it was created and
                % to delete any potential non-zip object which was
                % retreived.
                error(['zip file: ' zip_path ' cannot be found while attempting to download folder from XNAT. Are you sure the xnat select path was correct?']);
            end

            % Unzip file
            unzip(zip_path,xgp);

            % Remove zip file
            delete(zip_path);
        end 
        
        % Upload methods -------------------------------------------------%
        
        function upload_file(obj, xsp, xpfp)
            % Puts file in xpfp into xsp
            %
            % xpfp = xnat "put file" path
            
            obj.xnat.select(xsp).put(py.list({xpfp}),true); % true = overwrite contents
        end    
        
        function upload_folder(obj, xsp, xpfp)
            % Puts all the contents in xpfp into xsp
            %
            % xpfp = xnat "put folder" path
            
            obj.xnat.select(xsp).put_dir(xpfp,true); % true = overwrite contents
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