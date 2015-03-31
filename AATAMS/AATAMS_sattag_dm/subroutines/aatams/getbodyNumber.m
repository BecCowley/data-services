function bodyNumber = getbodyNumber (pttCode)

australianTagsFile = getenv('australian_tags_filepath');

%% read CSV file created by Xavier
aatamssattagmetadata = importAustralianTagCSV(australianTagsFile);

indexPTT   = str2double([aatamssattagmetadata.ptt]) == pttCode;
bodyNumber = char(aatamssattagmetadata.body(indexPTT));

end