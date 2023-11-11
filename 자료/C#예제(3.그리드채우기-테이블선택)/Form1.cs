using System;
using System.Data;
using System.Text;
using System.Windows.Forms;
using System.Data.OleDb;

namespace WindowsFormsApplication1
{
    public partial class Form1 : Form
    {
 
        OleDbConnection conn;
        string sql;
        public Form1()
        {
            InitializeComponent();
        }

         private void button1_Click(object sender, EventArgs e)
        {
            dataGridView1.Rows.Clear();

            sql = "Provider=MSDAORA;Password=" + txtBoxPw.Text + ";User ID=" + txtboxId.Text;//oracle 서버 연결
            //연결 스트링에 대한 정보 
            //Oracle - MSDAORA 
            //MS SQL - SQLOLEDB 
            //Data Source(server) : 데이터베이스 위치 
            //User ID/Password : 인증 정보 

            conn = new OleDbConnection(sql);
 

            try
            {
                conn.Open(); //데이터베이스 연결       
                //conn.Open(); //데이터베이스 연결
                OleDbCommand cmd = new OleDbCommand();
                cmd.CommandText = "select table_name from user_tables"; //테이블목록가져오기
                cmd.CommandType = CommandType.Text; //검색명령을 쿼리 형태로
                cmd.Connection = conn;

                OleDbDataReader read = cmd.ExecuteReader(); //select 결과

                //행 단위로 반복
                while (read.Read())
                {
                     cmbTableList.Items.Add(read.GetValue(0)); //데이터그리드뷰에 오브젝트 배열 추가
                }

                read.Close();
                cmbTableList.Text = "테이블선택";
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message); //에러 메세지 
            }
        }



         private void cmbTableList_SelectedIndexChanged(object sender, EventArgs e)
         {

             dataGridView1.Rows.Clear();

             sql = "Provider=MSDAORA;Password=" + txtBoxPw.Text + ";User ID=" + txtboxId.Text;//oracle 서버 연결

             conn = new OleDbConnection(sql);


             try
             {
                conn.Open(); //데이터베이스 연결         
                 //conn.Open(); //데이터베이스 연결
                 OleDbCommand cmd = new OleDbCommand();
                 cmd.CommandText = "select * from "+ cmbTableList.Text; 
                 cmd.CommandType = CommandType.Text;
                 cmd.Connection = conn;

                 OleDbDataReader read = cmd.ExecuteReader(); //select  결과
                 
                 dataGridView1.ColumnCount = read.FieldCount; //read.FieldCount는 테이블의 컬럼 수를 말함
                 //필드명 받아오는 반복문
                 for (int i = 0; i < read.FieldCount; i++)
                 {
                     dataGridView1.Columns[i].Name = read.GetName(i);
                 }

                 //행 단위로 반복
                 while (read.Read())
                 {
                     object[] obj = new object[read.FieldCount]; // 필드수만큼 오브젝트 배열

                     for (int i = 0; i < read.FieldCount; i++) // 필드 수만큼 반복
                     {
                         obj[i] = new object();
                         obj[i] = read.GetValue(i); // 오브젝트배열에 데이터 저장
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

 
         private void txtBoxPw_KeyDown(object sender, KeyEventArgs e)
         {
            if (e.KeyCode == Keys.Enter)
             {
                 button1_Click(sender, e);
                 e.SuppressKeyPress = true;
             }
         }

    }   
}

      