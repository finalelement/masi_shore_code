classdef xnat_utils < handle
    methods (Static, Access = public)
        function project_URI = get_project_URI(project)
            project_URI = fullfile(filesep,'projects',project);
        end
        
        function subject_URI = get_subject_URI(project, subject)
            subject_URI = fullfile(xnat_utils.get_project_URI(project),'subjects',subject);
        end
        
        function session_URI = get_session_URI(project, subject, session)
            session_URI = fullfile(xnat_utils.get_subject_URI(project,subject),'experiments',session);
        end
    end
end