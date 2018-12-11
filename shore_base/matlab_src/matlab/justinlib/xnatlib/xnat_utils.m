classdef xnat_utils < handle
    methods (Static, Access = public)
        function session_path = get_session_path(project, subject, session)    
            session_path = fullfile(filesep,'projects',project,'subjects',subject,'experiments',session);       
        end  
    end    
end