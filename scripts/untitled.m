archiveFolder = 'C:\Users\sijar\Downloads\TRUST_DR\datasets\archive';

allFiles = dir(fullfile(archiveFolder,'**','*'));
allFiles = allFiles(~[allFiles.isdir]);

for k = 1:numel(allFiles)
    if contains(lower(allFiles(k).name), ...
            ["csv","xls","xlsx","txt","json","label","annotation","ground"])
        fprintf('%s\n',fullfile(allFiles(k).folder,allFiles(k).name));
    end
end