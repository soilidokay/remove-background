pip install pillow
pip install rembg

pip install opencv-python numpy Pillow rembg

app9

py -m venv venv

Scaffold-DbContext "Data Source=sql01\dcsql;Initial Catalog=IT_ERPDB;User ID=erpuser;Password=Vf@ERP2018;TrustServerCertificate=True;Connect Timeout=60" Microsoft.EntityFrameworkCore.SqlServer -Tables "TblHRDCanteenMenu","TblHRDCanteenSpecialFood","TblHRDCanteenSpecialFoodForOT" -OutputDir "..\ERP.Domain\Entities" -ContextDir "Persistence" -Context "ERPDbContext"