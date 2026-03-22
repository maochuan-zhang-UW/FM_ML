function savemyfigureFM5_ML(name)
% SAVEMYFIGUREFM_FINAL Save current figure as PNG and PDF with optional name
%   If no name is provided, uses a timestamp (yyyymmdd_HHMMSS)

% Determine filename
if nargin < 1 || isempty(name)
    name = datestr(now, 'yyyymmdd_HHMMSS'); % Safe filename format
end

% Define paths
baseDir = '/Users/mczhang/Documents/GitHub/FM5_ML/03-output-graphics/';
pngPath = fullfile(baseDir, 'MyPng' , [name '.png']);
pdfPath = fullfile(baseDir, 'MyPdf', [name '.pdf']);

%Export graphics
exportgraphics(gcf, pngPath,...
    'Resolution', 300,...
    'BackgroundColor', 'white');

exportgraphics(gcf, pdfPath,...
    'ContentType', 'vector');
end
