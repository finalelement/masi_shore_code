classdef vol_utils < handle
% Library for commonly used volume (3D and 4D) utilities

    methods (Static, Access = public)    
        
        % 3D/4D related library ------------------------------------------%        
                       
        function vol = get_nifti_vol(nifti_path)            
            nifti = load_untouch_nii(nifti_path);
            vol = double(nifti.img); % Return as double precision            
        end
                
        function f = viewer_4D(data, figNum)
            % Simple viewer for 4D data; singleton dimensions will be
            % squeezed.
            
            % Squeeze data
            data = squeeze(data);
            
            % Prepare figure
            if exist('figNum','var')
                f = figure(figNum,'resize','off');
            else
                f = figure('resize','off');
            end
            
            clf(f);
            set(f,'Position',[100 100 500 500])
            ax = axes('Parent',f,'units','Pixels','Position',[100 100 300 300]);

            % Slider for 3rd dimension
            b1 = uicontrol('Parent',f,'Style','slider', ...
                'Position',[100 25 300 50],'Value',1,'min',1,'max',size(data,3), ...
                'Callback',@(es,ed) slider_callback());
            if size(data,3) == 1
                set(b1,'Enable','off');
            end

            t1 = uicontrol('Parent',f,'Style','text', ...
                'Position',[400 40 100 20],'String','text1');
            
            % Slider for 4th dimension
            b2 = uicontrol('Parent',f,'Style','slider', ...
                'Position',[25 100 50 300],'Value',1,'min',1,'max',size(data,4), ...
                'Callback',@(es,ed) slider_callback());
            if size(data,4) == 1
                set(b2,'Enable','off');
            end

            t2 = uicontrol('Parent',f,'Style','text', ...
                'Position',[0 440 100 20],'String','text2');

            % Initialize
            slider_callback();

            function slider_callback()  
                idx1 = 1;
                if size(data,3) ~= 1
                    idx1 = round(b1.Value);
                end

                idx2 = 1;
                if size(data,4) ~= 1
                    idx2 = round(b2.Value);
                end

                % Show slice
                imshow(data(:,:,idx1,idx2),[],'parent',ax)

                % Update text1
                set(t1,'String',[num2str(idx1) ' of ' num2str(size(data,3))])

                % Update text2
                set(t2,'String',[num2str(idx2) ' of ' num2str(size(data,4))])        
            end
        end           
    end    
end