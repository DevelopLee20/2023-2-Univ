using System;
using System.Data;
using System.Text;
using System.Windows.Forms;
using System.Data.OleDb;
using System.Data.Odbc;

namespace WindowsFormsApplication1
{
    public partial class Form1 : Form
    {
 
        OdbcConnection conn_odbc;
        OleDbConnection conn_ole;
        string sql;
        int Count;
        public Form1()
        {
            InitializeComponent();
        }
        private void button1_Click(object sender, EventArgs e)
        {
            string dsn;
            dataGridView1.Rows.Clear();
            dsn = Microsoft.VisualBasic.Interaction.InputBox("dsn를 입력하세요", "dsn입력");

            //======================================================================================
            sql = "dsn=" + dsn + ";PWD=" + txtBoxPw.Text + ";UID=" + txtboxId.Text;//oracle 서버 연결
           //=======================================================================================
            try
            {
                conn_odbc = new OdbcConnection(sql);
                conn_odbc.Open(); //데이터베이스 연결
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message); //에러 메세지 
            }

            try
            {
                OdbcCommand cmd = new OdbcCommand();
                cmd.CommandText = "select * from emp"; //member 테이블
                cmd.CommandType = CommandType.Text; //검색명령을 쿼리 형태로
                cmd.Connection = conn_odbc;
                OdbcDataReader read = cmd.ExecuteReader(); //select * from emp 결과
                Count = read.FieldCount;
                dataGridView1.ColumnCount = Count+1;
                //필드명 받아오는 반복문
                for (int i = 0; i < Count; i++)
                {
                    dataGridView1.Columns[i].Name = read.GetName(i);
                }

                //행 단위로 반복
                while (read.Read())
                {
                    
                 object[] obj = new object[Count]; // 필드수만큼 오브젝트 배열
                 for (int i = 0; i <Count; i++) // 필드 수만큼 반복
                 {
                                                           
                   obj[i] = new object();
                   obj[i] = read.GetValue(i); 
                  }
                                   
                                  
                 dataGridView1.Rows.Add(obj); //데이터그리드뷰에 오브젝트 배열 추가
                }

                read.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message); //에러 메세지 
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            dataGridView1.Rows.Clear();
            
            sql = "Provider=MSDAORA;Password=" + txtBoxPw.Text + ";User ID=" + txtboxId.Text;//oracle 서버 연결

            //연결 스트링에 대한 정보 
            //Oracle - MSDAORA 
            //MS SQL - SQLOLEDB 
            //Data Source(server) : 데이터베이스 위치 
            //User ID/Password : 인증 정보 

            conn_ole = new OleDbConnection(sql);
            conn_ole.Open(); //데이터베이스 연결         

            try
            {

                OleDbCommand cmd = new OleDbCommand();
                cmd.CommandText = "select * from emp"; 
                cmd.CommandType = CommandType.Text; //검색명령을 쿼리 형태로
                cmd.Connection = conn_ole;

                OleDbDataReader read = cmd.ExecuteReader(); //select * from emp 결과
               Count = read.FieldCount;
                dataGridView1.ColumnCount = Count+1;
                //필드명 받아오는 반복문
                for (int i = 0; i < Count; i++)
                {
                    dataGridView1.Columns[i].Name = read.GetName(i);
                }

                //행 단위로 반복
                while (read.Read())
                {
                    
                 object[] obj = new object[Count]; // 필드수만큼 오브젝트 배열
                 for (int i = 0; i <Count; i++) // 필드 수만큼 반복
                 {
                   obj[i] = new object();
                   obj[i] = read.GetValue(i); 
                  }
           
                 dataGridView1.Rows.Add(obj); //데이터그리드뷰에 오브젝트 배열 추가
                }

                read.Close();
            }
            
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message); //에러 메세지 
            }
        }

    }   
}

      