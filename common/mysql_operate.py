import pymysql
from config.setting import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWD, MYSQL_DB
import logging

logger = logging.getLogger('my_logger')

class MysqlDb():

    def __init__(self, host, port, user, passwd, db):
        self.conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            passwd=passwd,
            db=db,
            autocommit=True
        )
        self.cur = self.conn.cursor(cursor=pymysql.cursors.DictCursor)

    def __del__(self):
        self.cur.close()
        self.conn.close()

    def select_db(self, sql):
        self.conn.ping(reconnect=True)
        self.cur.execute(sql)
        data = self.cur.fetchall()
        print("SQL SELECT ====>> ",sql)
        logger.info("SQL SELECT ====>> %s", sql)
        return data

    def execute_db(self, sql,type):
        try:
            self.conn.ping(reconnect=True)
            self.cur.execute(sql)
            self.conn.commit()
            print("SQL execute ====>> ",sql)
            logger.info("SQL SELECT ====>> %s", sql)
            if type=='INSERT':   
                return {"status":0,"lastrowid":self.cur.lastrowid}
            elif type=='UPDATE':
                if self.cur.rowcount==0:
                    return {"status":-1,"err":"warning, check data"}
                else:
                    return {"status":0,"rowcount":self.cur.rowcount}
            else:
                return {"status":0}
        except Exception as e:
            print("error:{}".format(e))
            logger.error("SQL SELECT ====>> %s", sql)
            self.conn.rollback()
            return {"status":-1,"err":e}
        

db = MysqlDb(MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWD, MYSQL_DB)