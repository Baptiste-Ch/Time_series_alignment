"""Initial migration

Revision ID: 5060f18b083e
Revises: 
Create Date: 2023-08-01 11:52:14.857146

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5060f18b083e'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('counter_data')
    op.add_column('counter', sa.Column('max_counter', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('counter', 'max_counter')
    op.create_table('counter_data',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('counter', sa.VARCHAR(length=100), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###