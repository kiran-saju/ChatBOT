-- Create ChatBotDB and FAQ table
IF DB_ID('ChatBotDB') IS NULL
  CREATE DATABASE ChatBotDB;
GO
USE ChatBotDB;
GO

IF OBJECT_ID('dbo.faq','U') IS NULL
BEGIN
  CREATE TABLE dbo.faq(
    faq_id      INT IDENTITY(1,1) PRIMARY KEY,
    question    NVARCHAR(500) NOT NULL UNIQUE,
    answer_text NVARCHAR(MAX) NOT NULL,
    is_active   BIT NOT NULL CONSTRAINT DF_faq_active DEFAULT(1),
    created_at  DATETIME2(0) NOT NULL CONSTRAINT DF_faq_created DEFAULT (SYSUTCDATETIME()),
    updated_at  DATETIME2(0) NOT NULL CONSTRAINT DF_faq_updated DEFAULT (SYSUTCDATETIME())
  );
END
GO

IF OBJECT_ID('dbo.trg_faq_updated_at','TR') IS NULL
EXEC('
CREATE TRIGGER dbo.trg_faq_updated_at
ON dbo.faq
AFTER UPDATE
AS
BEGIN
  SET NOCOUNT ON;
  UPDATE f
     SET updated_at = SYSUTCDATETIME()
  FROM dbo.faq f
  JOIN inserted i ON i.faq_id = f.faq_id;
END
');
GO

-- (Optional) Seed initial FAQs
INSERT INTO dbo.faq (question, answer_text)
VALUES
(N'Free learning courses', N'We offer a set of free, self-paced courses. Start anytime.'),
(N'Connect to Counsellor', N'Our counsellor can guide you on course selection, fees, and eligibility.'),
(N'Are You Eligible? Find Out!', N'Share your highest qualification and years of experience for a quick check.'),
(N'Submit your Application', N'Apply online with your ID, CV and transcripts ready.');
GO


USE ChatBotDB;
SELECT * FROM dbo.faq;
