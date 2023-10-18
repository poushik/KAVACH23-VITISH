import psycopg2

import keys

conn = psycopg2.connect(
    host=keys.DATABASE_HOST,
    port=5432,
    database="cctv",
    user=keys.DATABASE_USER,
    password=keys.DATABASE_PASSWORD
)

# cur = conn.cursor()


# def saveAnamoly():
#     cur = conn.cursor()
#     cur.execute("""INSERT INTO Anomaly(
#         Serial_num, Camera_id, type, threat_level,street,city,pincode) VALUES
#         (%s,%s,%s,%s,%s,%s,%s)""",
#         ()
#         )

# cur.execute("SELECT * FROM Anomaly")
# rows = cur.fetchall()

# for row in rows:
#     print(row)

# cur.close()
# conn.close()






