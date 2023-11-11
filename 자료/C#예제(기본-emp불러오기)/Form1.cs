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
 
        OdbcConnection conn;
        OleDbConnection conn2;
        public Form1()
        {
            InitializeComponent();
        }
        private void button1_Click(object sender, EventArgs e)
        {
//C#에서는 InputBox가 없기 때문에 vb를 참조한다.
//1. 솔루션 탐색기안 참조 우클릭
//2. 참조 추가 선택
//3. Net 탭에서 Microsoft.VisualBasic 선택
//4. 확인 클릭

            string connString, id, pw, dsn;
            txtResult.Text = "";
            id= Microsoft.VisualBasic.Interaction.InputBox("아이디를 입력하세요", "ID입력");
            pw = Microsoft.VisualBasic.Interaction.InputBox("비밀번호를 입력하세요", "PW입력");
            dsn = Microsoft.VisualBasic.Interaction.InputBox("dsn를 입력하세요", "dsn입력");

            connString = "dsn=" + dsn + ";UID=" + id + ";PWD=" + pw; //oracle 서버 연결

            try
            {
                conn = new OdbcConnection(connString);
            conn.Open(); //데이터베이스 연결
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
                cmd.Connection = conn;
                OdbcDataReader read = cmd.ExecuteReader(); //select * from emp 결과

                //행 단위로 반복
                while (read.Read())
                {
                    object[] obj = new object[8]; // 필드수만큼 오브젝트 배열

                    for (int i = 0; i < 8; i++) // 필드 수만큼 반복
                    {
                        txtResult.AppendText(read.GetValue(i).ToString() + " / ");
                        obj[i] = new object();
                        obj[i] = read.GetValue(i); // 오브젝트배열에 데이터 저장
                    }
                    txtResult.AppendText("\r\n");

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
            string connString, id, pw;
            
            txtResult.Text = "";

            id = Microsoft.VisualBasic.Interaction.InputBox("아이디를 입력하세요", "ID입력");
            pw = Microsoft.VisualBasic.Interaction.InputBox("비밀번호를 입력하세요", "PW입력");
            connString = "Provider=OraOLEDB.Oracle;User ID=" + id + ";Password=" + pw; //oracle 서버 연결
            //연결 스트링에 대한 정보 
            //Oracle - MSDAORA 
            //MS SQL - SQLOLEDB 
            //User ID/Password : 인증 정보 
            conn2 = new OleDbConnection(connString);

            try
            {

                conn2.Open(); //데이터베이스 연결         
                OleDbCommand cmd = new OleDbCommand();
                cmd.CommandText = "select * from emp"; 
                cmd.CommandType = CommandType.Text; //검색명령을 쿼리 형태로
                cmd.Connection = conn2;

                OleDbDataReader read = cmd.ExecuteReader(); //select * from emp 결과
               
                //행 단위로 반복
                while (read.Read())
                {
                    object[] obj = new object[8]; // 필드수만큼 오브젝트 배열

                    for (int i = 0; i < 8; i++) // 필드 수만큼 반복
                    {
                        txtResult.AppendText(read.GetValue(i).ToString() + " / ");
                        obj[i] = new object();
                        obj[i] = read.GetValue(i); // 오브젝트배열에 데이터 저장
                    }
                    txtResult.AppendText("\r\n");

                  
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

      