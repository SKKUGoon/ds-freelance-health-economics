from pathlib import Path
import pandas as pd

def kcd_xlsx_to_parquet(
    xlsx_path: str | Path = "kcd.xlsx",
    parquet_path: str | Path = "kcd.parquet",
    *,
    sheet_name: str | int | None = "KCD-9 DB Masterfile",
) -> pd.DataFrame:
    """Read KCD Excel, keep key columns, and write to Parquet.

    Keeps only: 질병 분류코드, 한글명칭, 영문명칭.
    Returns the filtered dataframe.
    """
    xlsx_path = Path(xlsx_path)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Read raw sheet because the useful labels are not clean column headers.
    df_raw = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
    )

    # Fixed positions in KCD-9 DB Masterfile:
    # 2: 질병 분류코드, 5: 한글명칭, 6: 영문명칭
    df_kcd = df_raw.iloc[:, [2, 5, 6]].copy()
    df_kcd.columns = ["질병 분류코드", "한글명칭", "영문명칭"]

    # Remove title/header noise rows and empty code rows.
    df_kcd["질병 분류코드"] = df_kcd["질병 분류코드"].astype("string").str.strip()
    df_kcd = df_kcd[
        df_kcd["질병 분류코드"].notna()
        & (df_kcd["질병 분류코드"] != "")
        & (df_kcd["질병 분류코드"] != "질병분류코드")
    ].copy()

    # Normalize text columns to avoid parquet/pyarrow mixed-type issues.
    for col in df_kcd.columns:
        df_kcd[col] = df_kcd[col].map(
            lambda v: v.decode("utf-8", errors="replace")
            if isinstance(v, (bytes, bytearray))
            else v
        ).astype("string")

    df_kcd.iloc[1:].reset_index(drop=True).to_parquet(parquet_path, index=False)
    return df_kcd



if __name__ == "__main__":
    kcd_df = kcd_xlsx_to_parquet()
    print(kcd_df.head())
